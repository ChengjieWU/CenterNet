import os
import pickle
import json
import time
from collections import defaultdict
import datetime

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

from db.detection import DETECTION
from config import system_configs
from utils.visualize import display_instances
from .utils import compute_iou_matrix


category_trans_dict = {"拉链头0号": "LaLianTou",
                       "拉链头1号": "LaLianTou",
                       "拉链头2号": "LaLianTou",
                       "拉链头3号": "LaLianTou",
                       "拉链头4号": "LaLianTou",
                       "拉链头5号": "LaLianTou",
                       "锁扣头0号": "SuoKouTou",
                       "锁扣头1号": "SuoKouTou",
                       "锁扣头2号": "SuoKouTou",
                       "锁扣头3号": "SuoKouTou",
                       "皮签1": "PiQian",
                       "皮签2": "PiQian",
                       "铆钉0号": "MaoDing",
                       "铆钉1号": "MaoDing",
                       "铆钉2号": "MaoDing",
                       "铆钉3号": "MaoDing",
                       "产地标": "ChanDiBiao",
                       }

category_trans_dict_lv1 = {"五金-拉链头0": "LaLianTou",
                           "五金-拉链头1": "LaLianTou",
                           "五金-拉链头2": "LaLianTou",
                           "五金-拉链头3": "LaLianTou",
                           "五金-拉链头4": "LaLianTou",
                           "五金-拉链头5": "LaLianTou",
                           "五金-锁扣头0": "SuoKouTou",
                           "五金-锁扣头1": "SuoKouTou",
                           "五金-锁扣头2": "SuoKouTou",
                           "五金-锁扣头3": "SuoKouTou",
                           "皮签-皮签1": "PiQian",
                           "皮签-皮签2": "PiQian",
                           "五金-铆钉0": "MaoDing",
                           "五金-铆钉1": "MaoDing",
                           "五金-铆钉2": "MaoDing",
                           "五金-铆钉3": "MaoDing",
                           "出厂标号": "ChanDiBiao"}


class LV(DETECTION):
    def __init__(self, db_config, split="LV2", demo=False):
        """LV数据集格式

        :param db_config:
        :param split: "LV1", "LV2"
        :param demo: bool, if True, no data is actually loaded so split does not
                     matter.
        """
        super(LV, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split     # split名称
        self._dataset = {       # 实际使用的split名称
            "LV1": "LV1",
            "LV2": "LV2"
        }[self._split]
        self._data = "lv"       # 数据集名称

        self._LV_dir = os.path.join(data_dir, "lv")
        self._label_dir = os.path.join(self._LV_dir, "annotations", self._dataset)
        self._image_dir = os.path.join(self._LV_dir, "images", self._dataset)

        self._cat_ids = [
            "LaLianTou", "SuoKouTou", "PiQian", "MaoDing", "ChanDiBiao"]
        self._classes = {x+1: y for x, y in enumerate(self._cat_ids)}   # [1-5] -> _cat_id
        self._lv_to_class_map = {                 # 字典，_cat_id -> [1-5]
            value: key for key, value in self._classes.items()
        }

        if not demo:
            # self._detections是最主要的数据存储处
            # image file path -> [[x1, y1, x2, y2, [1-5]内部编号]]
            self._detections = None
            self._cache_file = os.path.join(cache_dir, "{}_{}.pkl".format(self._data, self._dataset))
            self._load_data()
            self._db_inds = np.arange(len(self._image_ids))     # 给所有图片统一的编号
            # self._image_ids与self._db_inds均是在BASE中定义的，self._image_ids保存所有
            # 图片的路径、或文件名，self._db_inds给所有图片统一编号

    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def class_name(self, cid):
        """使用[1-5]的内部类别编号，获取类别的拼音名字"""
        return self._classes[cid]

    def _image2annotation(self, image_path):
        """在LV数据集格式下，将image的路径转换为其标注的路径"""
        if self._dataset == "LV2":
            image_file = os.path.basename(image_path)
            bag_number = os.path.basename(os.path.dirname(image_path))
            label_string = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            return os.path.join(self._label_dir, label_string, bag_number, image_file[:-4] + ".json")
        elif self._dataset == "LV1":
            image_file = os.path.basename(image_path)
            bag_number = os.path.basename(os.path.dirname(image_path))
            label_string = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            type_string = os.path.basename(os.path.dirname(os.path.dirname(
                os.path.dirname(image_path))))
            return os.path.join(
                self._label_dir, type_string, label_string, bag_number,
                bag_number + "-" + image_file + ".json")
        else:
            raise NotImplementedError("other splits are not implemented")

    def _extract_data(self):
        """Extract data

        更新了self._image_ids, self._detections

        :return: None.
        """
        self._image_ids = list()    # image file path
        for dir_path, sub_dirs, files in os.walk(self._image_dir):
            for image_file in files:
                self._image_ids.append(os.path.join(dir_path, image_file))

        self._detections = {}

        # DEBUG
        # max_bbox = 0
        # max_bbox_id = 0

        for ind, image_id in enumerate(tqdm(self._image_ids)):
            bboxes = []
            categories = []
            annotation_path = self._image2annotation(image_id)
            if os.path.exists(annotation_path):
                with open(annotation_path, "r", encoding="utf-8") as fp:
                    json_dict = json.load(fp)
                if self._dataset == "LV2":
                    for item in json_dict["results"]:
                        bbox = np.array(item["bbox"])
                        bbox[[2, 3]] += bbox[[0, 1]]
                        bboxes.append(bbox)
                        categories.append(self._lv_to_class_map[
                                              category_trans_dict[item["class"]]])
                elif self._dataset == "LV1":
                    for item in json_dict["marks"]:
                        bbox = [item["point"]["xmin"], item["point"]["ymin"],
                                item["point"]["xmax"], item["point"]["ymax"]]
                        bbox = np.array(bbox)
                        bboxes.append(bbox)
                        categories.append(self._lv_to_class_map[
                            category_trans_dict_lv1[item["ptitle"]]])
                else:
                    raise NotImplementedError("other splits are not implemented")

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

        # DEBUG
        #     if len(categories) > max_bbox:
        #         max_bbox = len(categories)
        #         max_bbox_id = ind
        #
        # print(max_bbox, max_bbox_id, self._image_ids[max_bbox_id])
        # image_id = self._image_ids[max_bbox_id]
        # with Image.open(image_id) as fp:
        #     img = np.array(fp, dtype=np.uint8)
        # bboxes = self._detections[image_id]
        # display_instances(img, bboxes, ["background"] + self._cat_ids)

    def detections(self, ind):
        """使用初始全局编号，获取对应图片的detection"""
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]
        return detections.astype(float).copy()

    def image_file(self, ind):
        """使用初始全局编号，获取对应图片的文件路径"""
        # 此处覆盖BASE类中的实现，因为LV未使用到self._image_file属性
        return self._image_ids[ind]

    @staticmethod
    def _to_float(x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        """把all_bboxes的输出整理为COCO标注的格式

        :param all_bboxes: image_id -> {[1-5] -> (该类中检测到的数目, 5)},
                           分别为tl_xs, tl_ys, br_xs, br_ys, scores
        :return: detections: [{"image_id": , "category_id": , "bbox": , "score": }]
        """
        detections = []
        for image_id in all_bboxes:
            # coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        # "image_id": coco_id,
                        "image_id": image_id,       # 由于没有coco_id，此处使用image_id
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_detections(self, all_bboxes, score_threshold=0.):
        """把all_bboxes的输出整理为类detection的格式

        :param all_bboxes: image_id -> {[1-5] -> (该类中检测到的数目, 5)},
                           分别为tl_xs, tl_ys, br_xs, br_ys, scores
        :param score_threshold: float, 不接受score低于该阈值的候选框
        :return: detections: image_id -> [[x1, y1, x2, y2, [1-5], score]]
        """
        detections = {}
        for image_id in all_bboxes:
            detection = []
            for cls_ind in all_bboxes[image_id]:
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[4]
                    if score >= score_threshold:
                        bbox = list(map(self._to_float, bbox[0:4]))
                        bbox.extend([cls_ind, score])
                        detection.append(bbox)
            detection = np.array(detection, dtype=float)
            if detection.size == 0:
                detections[image_id] = np.zeros((0, 6), dtype=np.float32)
            else:
                detections[image_id] = detection
        return detections

    def evaluate(self, det_dt):
        eva = LVEval(self._detections, det_dt, self._cat_ids, self._classes)
        eva.evaluate()
        eva.accumulate()
        eva.summarize()
        return eva.stats

    def display(self, ind, save_path=None, show=True):
        """使用初始全局编号，显示对应图片，带有bounding box，用于显示训练集"""
        image_id = self._image_ids[ind]
        with Image.open(image_id) as fp:
            img = np.array(fp, dtype=np.uint8)
        bboxes = self._detections[image_id]
        display_instances(img, bboxes, ["background"] + self._cat_ids,
                          save_path=save_path, show=show)

    def display_detection(self, det, save_path=None, show=True):
        """输入convert_to_detections后产生的结果，显示所有图片，用于测试"""
        for image_id, bboxes in det.items():
            bboxes = np.array(bboxes, dtype=np.float32)
            with Image.open(image_id) as fp:
                img = np.array(fp, dtype=np.uint8)
            display_instances(img, bboxes[:, 0:5], ["background"] + self._cat_ids,
                              scores=bboxes[:, 5], save_path=save_path, show=show)

    def display_detection_demo(self, images, det, save_path=None, show=True):
        """输入convert_to_detections后产生的结果，显示所有图片，且输入图片，用于demo"""
        for image_id, bboxes in det.items():
            bboxes = np.array(bboxes, dtype=np.float32)
            img = images[image_id]
            # 模型中使用BGR，而visualize中使用RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display_instances(img, bboxes[:, 0:5],
                              ["background"] + self._cat_ids,
                              scores=bboxes[:, 5], save_path=save_path,
                              show=show)


class LVEval:
    def __init__(self, det_gt, det_dt, catIDs, classes):
        """LV数据集上的评估器

        :param det_gt: {image_id -> [[x1, y1, x2, y2, [1-5]内部编号]]}
        :param det_dt: {image_id -> [[x1, y1, x2, y2, [1-5]内部编号, score]]}
        :param catIDs: a list of category ids
        :param classes: {[1-5]内部编号 -> cat_id}
        """
        self.det_dt = det_dt
        self.imageIDs = list(self.det_dt.keys())  # list of image_id
        self.det_gt = dict()
        for image_id in self.imageIDs:
            self.det_gt[image_id] = det_gt[image_id]

        self.maxDets = [1, 10, 100]
        self.iouThrs = np.linspace(
            .5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1,
                                   endpoint=True)

        self.catIDs = catIDs       # list of cat_id
        self.classes = classes

        # 准备数据
        # {(image_id, cat_id) -> [[x1, y1, x2, y2]]}: defaultdict(list)
        self._gts = defaultdict(list)
        # {(image_id, cat_id) -> [[x1, y1, x2, y2, score]]}: defaultdict(list)
        self._dts = defaultdict(list)

        self.ious = None
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self.stats = None

    def _prepare(self):
        """生成self._gts与self._dts，将self.evalImgs与self.eval初始化

        :return: None
        """
        # self._gts: {(image_id, cat_id) -> [[x1, y1, x2, y2]]}: defaultdict(list)
        # self._dts: {(image_id, cat_id) -> [[x1, y1, x2, y2, score]]}: defaultdict(list)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for image_id, bboxes in self.det_gt.items():
            for bbox in bboxes:
                self._gts[image_id, self.classes[int(bbox[4])]].append(bbox[0:4])
        for image_id, bboxes in self.det_dt.items():
            for bbox in bboxes:
                self._dts[image_id, self.classes[int(bbox[4])]].append((bbox[[0, 1, 2, 3, 5]]))

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval = {}                  # accumulated evaluation results

    def evaluate(self):
        tic = time.time()
        print('Running per image evaluation...')

        self._prepare()     # 生成self._gts与self._dts，将self.evalImgs与self.eval初始化

        # 将self._dts按照分数从高到低进行排序，且已经保证最多只有maxDets[-1]个框
        for image_id in self.imageIDs:
            for cat_id in self.catIDs:
                dt = self._dts[image_id, cat_id]
                inds = np.argsort([-d[4] for d in dt], kind='mergesort')
                dt = [dt[i] for i in inds]
                if len(dt) > self.maxDets[-1]:
                    dt = dt[0:self.maxDets[-1]]
                self._dts[image_id, cat_id] = dt

        # self.ious: {(image_id, cat_id) -> IoU matrix}
        # 其中第i行第j列表示第i个d与第j个g的IoU
        self.ious = {(image_id, cat_id): self.computeIoU(image_id, cat_id)
                     for image_id in self.imageIDs for cat_id in self.catIDs}
        # self.evalImgs: [{"image_id": , "cat_id": , "dtMatches": , "gtMatches": , "dtScores": }]
        # 其中，dtMatches的shape为(T, D)，gtMatches的shape为(T, G)，其中T是阈值的数目，D为dt检测框数目，G为gt检测框数目
        # 且这个list是按照先catId，后imgId的顺序的。
        # 如果对一个图，一个类别，g与d都为空，则相应的值为None。
        self.evalImgs = [
            self.evaluateImg(imgId, catId)
            for catId in self.catIDs for imgId in self.imageIDs
        ]

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, image_id, cat_id):
        gt = self._gts[image_id, cat_id]
        dt = self._dts[image_id, cat_id]
        if len(gt) == 0 and len(dt) == 0:
            return []

        d = [d[0:4] for d in dt]
        ious = compute_iou_matrix(d, gt)
        return ious

    def evaluateImg(self, image_id, cat_id):
        """Perform evaluation for single category and image

        匹配原则：从最高分的d开始，匹配iou最大的g。若小于阈值，则匹配不到。

        :return: dict (single image results)
        """
        gt = self._gts[image_id, cat_id]
        dt = self._dts[image_id, cat_id]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # load computed ious
        ious = self.ious[image_id, cat_id]

        T = len(self.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(self.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, continue
                        if gtm[tind, gind] > 0:
                            continue
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtm[tind, dind] = m + 1
                    gtm[tind, m] = dind + 1
        # store results for given image and category
        return {
                'image_id':     image_id,
                'category_id':  cat_id,
                'dtMatches':    dtm,        # (T, D)
                'gtMatches':    gtm,        # (T, G)
                'dtScores':     [d[4] for d in dt],     # (D, )
            }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in self.eval

        :param p: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters

        T = len(self.iouThrs)
        R = len(self.recThrs)
        K = len(self.catIDs)
        A = 1
        M = len(self.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        # get inds to evaluate
        k_list = list(range(len(self.catIDs)))
        m_list = self.maxDets[:]
        a_list = list(range(1))
        i_list = list(range(len(self.imageIDs)))

        I0 = len(self.imageIDs)
        A0 = 1
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    # E: 当前类别、area下，所有image的evalImg的list
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    # dtScores: 所有image的dtScores首尾相接成一个列表
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    # dtm: 第一维大小为T，第二维大小与dtScores相同
                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]

                    npig = np.sum([e["gtMatches"].shape[1] for e in E])

                    tps = np.logical_and(dtm, True)     # true positive
                    fps = np.logical_not(dtm)           # false positive

                    # tp_sum, fp_sum: 第一维大小为T，第二维大小与dtScores相同
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)        # 所有检测框数目
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))     # tp / (fp + tp)
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        # 若检测到该类物品，则计算recall值
                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        # 反之，强行将该类的recall置为0
                        else:
                            recall[t, k, a, m] = 0

                        # Currently, we use this simple version of precision
                        if nd:
                            precision[t, :, k, a, m] = np.full(R, pr[-1])
                        else:
                            precision[t, :, k, a, m] = np.full(R, 1)

                        # TODO: implement both precision
                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        # pr = pr.tolist()
                        # q = q.tolist()
                        #
                        # for i in range(nd - 1, 0, -1):
                        #     if pr[i] > pr[i-1]:
                        #         pr[i-1] = pr[i]
                        #
                        # inds = np.searchsorted(rc, self.recThrs, side='left')
                        # try:
                        #     for ri, pi in enumerate(inds):
                        #         q[ri] = pr[pi]
                        #         ss[ri] = dtScoresSorted[pi]
                        # except:
                        #     pass
                        # precision[t, :, k, a, m] = np.array(q)
                        # scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            # 'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note this functin can *only* be applied on the default parameter setting
        """
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            assert areaRng == "all", "Not implemented for sizes"

            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(['all']) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((6,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.maxDets[2])
            stats[3] = _summarize(0, maxDets=self.maxDets[0])
            stats[4] = _summarize(0, maxDets=self.maxDets[1])
            stats[5] = _summarize(0, maxDets=self.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        self.stats = _summarizeDets()
