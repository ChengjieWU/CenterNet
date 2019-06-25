import pdb
import torch
import torch.nn as nn

from .utils import convolution, residual

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_ct_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):
    # 从feat中，取出所需点的feature vector
    # feat: (batch_size, feature_size[0] * feature_size[1], feature_channels)
    # ind: (batch_size, max_tag_len)
    dim = feat.size(2)
    # ind: (batch_size, max_tag_len, feature_channels)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # feat: (batch_size, max_tag_len, feature_channels)
    feat = feat.gather(1, ind)
    if mask is not None:
        # mask: (batch_size, max_tag_len, feature_channels)
        # 若有tag，则为1，否则为0
        mask = mask.unsqueeze(2).expand_as(feat)
        # feat: (batch_size, tag_len, feature_channels)
        feat = feat[mask]
        # feat: (batch_size * tag_len, feature_channels)
        # 估计只有在test的时候，mask才不置为None
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    """输入heatmap，返回K个值最大（每幅图全局最大，不区分类别）的点"""
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    # 返回的全是(batch, K)大小的array
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    """把网络的输出（8个feature map）解码为detections和center

    :return: center: (batch, K, 4), 分别为ct_xs, ct_ys, ct_clses, ct_scores
             detections: (batch, num_dets, 8), 分别为tl_xs, tl_ys, br_xs, br_ys,
                         scores, tl_scores, br_scores, clses
             需要注意的是，此处的坐标都是最后的feature map的大小中的坐标！！！
    """
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    ct_heat = torch.sigmoid(ct_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    ct_heat = _nms(ct_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)   # (batch, K, 2)
        tl_regr = tl_regr.view(batch, K, 1, 2)                  # (batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)   # (batch, K, 2)
        br_regr = br_regr.view(batch, 1, K, 2)                  # (batch, K, 1, 2)
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)   # (batch, K, 2)
        ct_regr = ct_regr.view(batch, 1, K, 2)                  # (batch, K, 1, 2)

        # 使用tl_regr、br_regr、ct_regr修正结果
        tl_xs = tl_xs + tl_regr[..., 0]     # (batch, K, K), (i, j)固定i对任意j值相同
        tl_ys = tl_ys + tl_regr[..., 1]     # (batch, K, K), (i, j)固定i对任意j值相同
        br_xs = br_xs + br_regr[..., 0]     # (batch, K, K), (i, j)固定j对任意i值相同
        br_ys = br_ys + br_regr[..., 1]     # (batch, K, K), (i, j)固定j对任意i值相同
        ct_xs = ct_xs + ct_regr[..., 0]     # (batch, K, K), (i, j)固定j对任意i值相同
        ct_ys = ct_ys + ct_regr[..., 1]     # (batch, K, K), (i, j)固定j对任意i值相同

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)   # (batch, K, K, 4)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)         # (batch, K, 1)
    tl_tag = tl_tag.view(batch, K, 1)                           # (batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)         # (batch, K, 1)
    br_tag = br_tag.view(batch, 1, K)                           # (batch, 1, K)
    # 计算任两对点的embedding值的距离
    # dists[0][i][j] = torch.abs(tl_tag[0][i][0] - br_tag[0][0][j])
    dists  = torch.abs(tl_tag - br_tag)                         # (batch, K, K)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)     # (batch, K, K), (i, j)固定i对任意j值相同
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)     # (batch, K, K), (i, j)固定j对任意i值相同
    scores    = (tl_scores + br_scores) / 2     # (batch, K, K), (i, j)表示i号tl与j号br的平均score

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)   # (batch, K, K), (i, j)固定i对任意j值相同
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)   # (batch, K, K), (i, j)固定j对任意i值相同
    cls_inds = (tl_clses != br_clses)   # (batch, K, K), (i, j) = False iff 第i个tl与第j个br属于同一类

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)  # (batch, K, K), (i, j) = False iff 第i个tl与第j个br的embedding差距小于ae_threshold

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)       # (batch, K, K), (i, j) = False iff tl_xi <= br_xj
    height_inds = (br_ys < tl_ys)       # (batch, K, K), (i, j) = False iff tl_yi <= br_yj

    # 使用classes、distances、widths and heights共3个条件筛去不符合要求的所有点对
    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)     # (batch, K * K)
    scores, inds = torch.topk(scores, num_dets)     # 选择平均分最高的num_dets个点对，返回(batch, num_dets)
    scores = scores.unsqueeze(2)        # (batch, num_dets, 1)

    # 使用上面的条件，最终在K * K对候选中，选出num_dets对
    bboxes = bboxes.view(batch, -1, 4)      # (batch, K * K, 4)
    bboxes = _gather_feat(bboxes, inds)     # (batch, num_dets, 4)
    
    #width = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    #height = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    
    clses  = tl_clses.contiguous().view(batch, -1, 1)   # (batch, K * K, 1)，K个连续的数是一样的
    clses  = _gather_feat(clses, inds).float()          # (batch, num_dets, 1)

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)   # (batch, K * K, 1)，K个连续的数是一样的
    tl_scores = _gather_feat(tl_scores, inds).float()       # (batch, num_dets, 1)
    br_scores = br_scores.contiguous().view(batch, -1, 1)   # (batch, K * K, 1)，K个数的序列重复K次
    br_scores = _gather_feat(br_scores, inds).float()       # (batch, num_dets, 1)

    ct_xs = ct_xs[:,0,:]    # (batch, K)
    ct_ys = ct_ys[:,0,:]    # (batch, K)

    # center: (batch, K, 4), 分别为ct_xs, ct_ys, ct_clses, ct_scores
    center = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), ct_scores.unsqueeze(2)], dim=2)
    # detections: (batch, num_dets, 8), 分别为tl_xs, tl_ys, br_xs, br_ys, scores, tl_scores, br_scores, clses
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections, center

def _neg_loss(preds, gt):
    """用作FOCAL loss"""
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    """带有clamp的sigmoid"""
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    """计算AE loss

    使用mask来确定有哪些标签，即确定一幅图中物品的数目。tag0与tag1，先在图中回归出每一个点
    的embedding，随后用ground truth的物品点位，选出每个物品的成对的两个点。训练时，想要来
    自相同物品的两点距离小，来自不同物品两点距离大。

    :param tag0: top left embedding, (batch_size, max_tag_len, feature_channels)
    :param tag1: bottom right embedding, (batch_size, max_tag_len, feature_channels)
    :param mask: ground truth mask, (batch_size, max_tag_len). 为1若标签存在。
    :return:
    """
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    """计算offset loss

    对图中每个像素，回归出offset，并且已经用ground truth的物品点位取出所需点位的offset，
    与ground truth的offset计算loss。

    :param regr: (batch_size, max_tag_len, 2)
    :param gt_regr: (batch_size, max_tag_len, 2)
    :param mask: ground truth mask, (batch_size, max_tag_len). 为1若标签存在。
    :return:
    """
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
