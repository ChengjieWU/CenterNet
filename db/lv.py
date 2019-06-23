import os
import pickle
import json

from tqdm import tqdm
import numpy as np
from PIL import Image

from db.detection import DETECTION
from config import system_configs
from utils.visualize import display_instances


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

category_zh_dict = {
    "LaLianTou": "拉链头",
    "SuoKouTou": "锁扣头",
    "PiQian": "皮签",
    "MaoDing": "铆钉",
    "ChanDiBiao": "产地标",
}


class LV(DETECTION):
    def __init__(self, db_config):
        super(LV, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._LV_dir = os.path.join(data_dir, "lv")
        self._label_dir = os.path.join(self._LV_dir, "annotations")
        self._image_dir = os.path.join(self._LV_dir, "images")

        self._data = "lv"

        self._cat_ids = [
            "LaLianTou", "SuoKouTou", "PiQian", "MaoDing", "ChanDiBiao"]
        self._classes = {x+1: y for x, y in enumerate(self._cat_ids)}
        self._lv_to_class_map = {                 # 字典，_cat_id -> [1-5]
            value: key for key, value in self._classes.items()
        }

        # self._detections是最主要的数据存储处
        # image file path -> [[x1, y1, x2, y2, cat_id]]
        self._detections = None
        self._cache_file = os.path.join(cache_dir, "{}.pkl".format(self._data))
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
        """使用[1-5]的内部类别编号，获取类别的中文名字"""
        return category_zh_dict[self._classes[cid]]

    def _image2annotation(self, image_path):
        """在LV数据集格式下，将image的路径转换为其标注的路径"""
        image_file = os.path.basename(image_path)
        bag_number = os.path.basename(os.path.dirname(image_path))
        label_string = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        return os.path.join(self._label_dir, label_string, bag_number, image_file[:-4] + ".json")

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
        for ind, image_id in enumerate(tqdm(self._image_ids)):
            bboxes = []
            categories = []
            annotation_path = self._image2annotation(image_id)
            if os.path.exists(annotation_path):
                with open(annotation_path, "r", encoding="utf-8") as fp:
                    json_dict = json.load(fp)
                for item in json_dict["results"]:
                    bbox = np.array(item["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)
                    categories.append(self._lv_to_class_map[
                                          category_trans_dict[item["class"]]])

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        """使用初始全局编号，获取对应图片的detection"""
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]
        return detections.astype(float).copy()

    def image_file(self, ind):
        """使用初始全局编号，获取对应图片的文件路径"""
        # 此处覆盖BASE类中的实现，因为LV未使用到self._image_file属性
        return self._image_ids[ind]

    def display(self, ind):
        """使用此打乱时刻的全局编号，显示对应图片，带有bounding box"""
        image_id = self._image_ids[self._db_inds[ind]]
        with Image.open(image_id) as fp:
            img = np.array(fp, dtype=np.uint8)
        bboxes = self._detections[image_id]
        display_instances(img, bboxes, ["background"] + self._cat_ids)
