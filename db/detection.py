import numpy as np
from db.base import BASE

class DETECTION(BASE):
    def __init__(self, db_config):
        super(DETECTION, self).__init__()

        self._configs["categories"]      = 80       # 数据中类别的数目
        self._configs["kp_categories"]   = 1        # 似乎从来没有使用到
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2

        # 网络与输入输出大小无关，可处理任意大小的图片，只是训练时，设定如下
        self._configs["input_size"]      = [511]    # 网络的输入图片大小，由于全卷积，故建立模型中用不到，实际只在sample准备训练数据图片时使用到
        self._configs["output_sizes"]    = [[128, 128]]  # 网络输出图片大小，由于全卷积，故建立模型中用不到，实际只在sample准备训练数据标注时使用到

        self._configs["nms_threshold"]   = 0.5      # soft_nms中的threshold
        self._configs["max_per_image"]   = 100      # 检测时，每张图片bbox的最大数目
        self._configs["top_k"]           = 100      # 选择K个值最大的点
        self._configs["ae_threshold"]    = 0.5      # embedding的距离大于此，即判定不是同一个物体的框
        self._configs["nms_kernel"]      = 3        # nms kernel size

        self._configs["nms_algorithm"]   = "exp_soft_nms"   # {"nms": 0, "linear_soft_nms": 1, "exp_soft_nms": 2}
        self._configs["weight_exp"]      = 8        # weight_exp in soft_nms
        self._configs["merge_bbox"]      = False    # bool, True时使用soft_nms_merge, 否则使用soft_nms
        
        self._configs["data_aug"]        = True     # bool
        self._configs["lighting"]        = True     # bool

        self._configs["border"]          = 128
        self._configs["gaussian_bump"]   = True
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_pushes"]     = False
        self._configs["rand_samples"]    = False
        self._configs["special_crop"]    = False

        self._configs["test_scales"]     = [1]      # 浮点数，使用哪些scale进行测试

        # 下面这三条似乎也从未使用到
        self._train_cfg["rcnn"] = dict(
                            assigner=dict(
                                pos_iou_thr=0.5,
                                neg_iou_thr=0.5,
                                min_pos_iou=0.5,
                                ignore_iof_thr=-1),
                            sampler=dict(
                                num=512,
                                pos_fraction=0.25,
                                neg_pos_ub=-1,
                                add_gt_as_proposals=True,
                                pos_balance_sampling=False,
                                neg_balance_thr=0),
                            mask_size=28,
                            pos_weight=-1,
                            debug=False)

        self._model['bbox_roi_extractor'] = dict(
                            type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                            out_channels=256,
                            featmap_strides=[4])

        self._model['bbox_head'] = dict(
                            type='SharedFCBBoxHead',
                            num_fcs=2,
                            in_channels=256,
                            fc_out_channels=1024,
                            roi_feat_size=7,
                            num_classes=81,
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.2, 0.2],
                            reg_class_agnostic=False)

        self.update_config(db_config)

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )
