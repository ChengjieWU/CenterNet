import os
import cv2
import json
import copy
import numpy as np
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge
from .coco import _rescale_dets, kp_decode
from db.lv import LV
from nnet.py_factory import NetworkFactory

colours = np.random.rand(5, 3)      # 一共有5类，故颜色为(5, 3)


def kp_detection(db: LV, nnet: NetworkFactory,
                 result_dir, debug=False, decode_func=kp_decode):
    """即检测模型的testing函数，在lv_test.py中被直接调用。

    :param db: dataset
    :param nnet: a NetworkFactory object
    :param result_dir: result directory path
    :param debug: bool
    :param decode_func:
    :return:
    """
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    db_inds = db.db_inds[100:200] if debug else db.db_inds[:5000]  # 取5000张图片
    num_images = db_inds.size

    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel = db.configs["nms_kernel"]

    scales = db.configs["test_scales"]
    weight_exp = db.configs["weight_exp"]
    merge_bbox = db.configs["merge_bbox"]
    categories = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id = db.image_ids(db_ind)
        image_file = db.image_file(db_ind)
        image = cv2.imread(image_file)

        height, width = image.shape[0:2]

        detections = []
        center_points = []

        for scale in scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            # 不懂为什么要做这个按位或
            inp_height = new_height | 127
            inp_width = new_width | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            # (inp_height + 1)、(inp_width + 1)肯定可以被4除尽
            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            # 先按照scale来resize
            resized_image = cv2.resize(image, (new_width, new_height))
            # 然后使用scale后的image的中心点，与inp_height、inp_width进行crop
            # 由于inp_height、inp_width一定是比new_height、new_width大的，故这一步
            # 实际上是在按照中心，扩大图片，并在周围补黑边。
            resized_image, border, offset = crop_image(
                resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)

            # resized_image是(H, W, C)，现在改成(C, H, W)以供pytorch使用
            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            # 这个size是有内容的图片大小，resized_image的大小为[inp_height, inp_width]
            sizes[0] = [int(height * scale), int(width * scale)]
            # 这个是out比上inp
            ratios[0] = [height_ratio, width_ratio]

            # 这个似乎是把原图和垂直翻折后的图片放在一起
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            # dets: (batch, 2 * num_dets, 8)
            # center: (batch, 2 * K, 4)
            dets, center = decode_func(nnet, images, K,
                                       ae_threshold=ae_threshold,
                                       kernel=nms_kernel)
            dets = dets.reshape(2, -1, 8)
            center = center.reshape(2, -1, 4)
            # 这两步是把垂直翻折后图片的检测结果，变换到原图上
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            center[1, :, [0]] = out_width - center[1, :, [0]]
            dets = dets.reshape(1, -1, 8)       # (1, 2 * num_dets, 8)
            center = center.reshape(1, -1, 4)   # (1, 2 * K, 4)

            # 去除在原图中不合法的框
            _rescale_dets(dets, ratios, borders, sizes)

            center[..., [0]] /= ratios[:, 1][:, None, None]
            center[..., [1]] /= ratios[:, 0][:, None, None]
            center[..., [0]] -= borders[:, 2][:, None, None]
            center[..., [1]] -= borders[:, 0][:, None, None]
            np.clip(center[..., [0]], 0, sizes[:, 1][:, None, None],
                    out=center[..., [0]])
            np.clip(center[..., [1]], 0, sizes[:, 0][:, None, None],
                    out=center[..., [1]])

            # 回复到原图中的坐标
            dets[:, :, 0:4] /= scale
            center[:, :, 0:2] /= scale

            # center point只使用scale为1的时候
            if scale == 1:
                center_points.append(center)
            detections.append(dets)

        # 把所有scale下检测出的统一合并起来
        detections = np.concatenate(detections, axis=1)         # (1, 2 * num_dets * len(scales), 8)
        center_points = np.concatenate(center_points, axis=1)   # (1, 2 * K, 4)

        classes = detections[..., -1]
        classes = classes[0]            # (2 * num_dets * len(scales),)
        detections = detections[0]      # (2 * num_dets * len(scales), 8)
        center_points = center_points[0]    # (2 * K, 4)

        # 获得所有的合法候选框
        valid_ind = detections[:, 4] > -1
        valid_detections = detections[valid_ind]    # (合法候选框, 8)

        box_width = valid_detections[:, 2] - valid_detections[:, 0]     # (合法候选框,)
        box_height = valid_detections[:, 3] - valid_detections[:, 1]    # (合法候选框,)

        # 小候选框与大候选框
        s_ind = (box_width * box_height <= 22500)
        l_ind = (box_width * box_height > 22500)

        s_detections = valid_detections[s_ind]  # (小框, 8)
        l_detections = valid_detections[l_ind]  # (大框, 8)

        # 小框：判断中心区域是否有中心点
        # 只要中心区域有一个同类中心点即可，分数按最高的算
        s_left_x = (2 * s_detections[:, 0] + s_detections[:, 2]) / 3
        s_right_x = (s_detections[:, 0] + 2 * s_detections[:, 2]) / 3
        s_top_y = (2 * s_detections[:, 1] + s_detections[:, 3]) / 3
        s_bottom_y = (s_detections[:, 1] + 2 * s_detections[:, 3]) / 3

        s_temp_score = copy.copy(s_detections[:, 4])
        s_detections[:, 4] = -1

        center_x = center_points[:, 0][:, np.newaxis]
        center_y = center_points[:, 1][:, np.newaxis]
        s_left_x = s_left_x[np.newaxis, :]
        s_right_x = s_right_x[np.newaxis, :]
        s_top_y = s_top_y[np.newaxis, :]
        s_bottom_y = s_bottom_y[np.newaxis, :]

        ind_lx = (center_x - s_left_x) > 0
        ind_rx = (center_x - s_right_x) < 0
        ind_ty = (center_y - s_top_y) > 0
        ind_by = (center_y - s_bottom_y) < 0
        ind_cls = (center_points[:, 2][:, np.newaxis] - s_detections[:, -1][np.newaxis, :]) == 0
        ind_s_new_score = np.max(((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0)), axis=0) == 1
        index_s_new_score = np.argmax(
            ((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) & (ind_cls + 0))[:, ind_s_new_score],
            axis=0)
        s_detections[:, 4][ind_s_new_score] = \
            (s_temp_score[ind_s_new_score] * 2 + center_points[index_s_new_score, 3]) / 3

        # 大框：判断中心区域是否有中心点
        l_left_x = (3 * l_detections[:, 0] + 2 * l_detections[:, 2]) / 5
        l_right_x = (2 * l_detections[:, 0] + 3 * l_detections[:, 2]) / 5
        l_top_y = (3 * l_detections[:, 1] + 2 * l_detections[:, 3]) / 5
        l_bottom_y = (2 * l_detections[:, 1] + 3 * l_detections[:, 3]) / 5

        l_temp_score = copy.copy(l_detections[:, 4])
        l_detections[:, 4] = -1

        center_x = center_points[:, 0][:, np.newaxis]
        center_y = center_points[:, 1][:, np.newaxis]
        l_left_x = l_left_x[np.newaxis, :]
        l_right_x = l_right_x[np.newaxis, :]
        l_top_y = l_top_y[np.newaxis, :]
        l_bottom_y = l_bottom_y[np.newaxis, :]

        ind_lx = (center_x - l_left_x) > 0
        ind_rx = (center_x - l_right_x) < 0
        ind_ty = (center_y - l_top_y) > 0
        ind_by = (center_y - l_bottom_y) < 0
        ind_cls = (center_points[:, 2][:, np.newaxis] - l_detections[:, -1][
                                                        np.newaxis, :]) == 0
        ind_l_new_score = np.max(
            ((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) &
             (ind_by + 0) & (ind_cls + 0)),
            axis=0) == 1
        index_l_new_score = np.argmax(
            ((ind_lx + 0) & (ind_rx + 0) & (ind_ty + 0) & (ind_by + 0) &
             (ind_cls + 0))[:, ind_l_new_score],
            axis=0)
        l_detections[:, 4][ind_l_new_score] = \
            (l_temp_score[ind_l_new_score] * 2 + center_points[index_l_new_score, 3]) / 3

        # 合并大框小框的检测结果，并按照score排序
        detections = np.concatenate([l_detections, s_detections], axis=0)
        detections = detections[np.argsort(-detections[:, 4])]
        classes = detections[..., -1]

        # for i in range(detections.shape[0]):
        #   box_width = detections[i,2]-detections[i,0]
        #   box_height = detections[i,3]-detections[i,1]
        #   if box_width*box_height<=22500 and detections[i,4]!=-1:
        #     left_x = (2*detections[i,0]+1*detections[i,2])/3
        #     right_x = (1*detections[i,0]+2*detections[i,2])/3
        #     top_y = (2*detections[i,1]+1*detections[i,3])/3
        #     bottom_y = (1*detections[i,1]+2*detections[i,3])/3
        #     temp_score = copy.copy(detections[i,4])
        #     detections[i,4] = -1
        #     for j in range(center_points.shape[0]):
        #        if (classes[i] == center_points[j,2])and \
        #           (center_points[j,0]>left_x and center_points[j,0]< right_x) and \
        #           ((center_points[j,1]>top_y and center_points[j,1]< bottom_y)):
        #           detections[i,4] = (temp_score*2 + center_points[j,3])/3
        #           break
        #   elif box_width*box_height > 22500 and detections[i,4]!=-1:
        #     left_x = (3*detections[i,0]+2*detections[i,2])/5
        #     right_x = (2*detections[i,0]+3*detections[i,2])/5
        #     top_y = (3*detections[i,1]+2*detections[i,3])/5
        #     bottom_y = (2*detections[i,1]+3*detections[i,3])/5
        #     temp_score = copy.copy(detections[i,4])
        #     detections[i,4] = -1
        #     for j in range(center_points.shape[0]):
        #        if (classes[i] == center_points[j,2])and \
        #           (center_points[j,0]>left_x and center_points[j,0]< right_x) and \
        #           ((center_points[j,1]>top_y and center_points[j,1]< bottom_y)):
        #           detections[i,4] = (temp_score*2 + center_points[j,3])/3
        #           break

        # reject detections with negative scores
        keep_inds = (detections[:, 4] > -1)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold,
                               method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold,
                         method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1]
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if debug:
            image_file = db.image_file(db_ind)
            image = cv2.imread(image_file)
            im = image[:, :, (2, 1, 0)]
            fig, ax = plt.subplots(figsize=(12, 12))
            fig = ax.imshow(im, aspect='equal')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # bboxes = {}
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.4)
                cat_name = db.class_name(j)
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    score = bbox[4]
                    bbox = bbox[0:4].astype(np.int32)
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[2]
                    ymax = bbox[3]
                    # if (xmax - xmin) * (ymax - ymin) > 5184:
                    ax.add_patch(
                        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                      fill=False, edgecolor=colours[j - 1],
                                      linewidth=4.0))
                    ax.text(xmin + 1, ymin - 3, '{} {:.3f}'.format(cat_name, score),
                            bbox=dict(facecolor=colours[j - 1], ec='black',
                                      lw=2, alpha=0.5),
                            fontsize=15, color='white', weight='bold')

            # debug_file1 = os.path.join(debug_dir, "{}.pdf".format(db_ind))
            debug_file2 = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            # plt.savefig(debug_file1)
            plt.savefig(debug_file2)
            plt.close()
            # cv2.imwrite(debug_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # 同时保存gt图以供对比
            db.display(db_ind, os.path.join(debug_dir, "{}_gt.jpg".format(db_ind)), show=False)

    # top_bboxes: image_id -> {[1-5] -> (该类中检测到的数目, 5)}, 分别为tl_xs, tl_ys, br_xs, br_ys, scores
    # 显示检测结果
    detections = db.convert_to_detections(top_bboxes, score_threshold=0.5)
    # db.display_detection(detections)
    # 评估检测结果
    db.evaluate(detections)
    return 0


def testing(db, nnet, result_dir, debug=False):
    return globals()[system_configs.sampling_function](db, nnet, result_dir,
                                                       debug=debug)
