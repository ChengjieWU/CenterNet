import os
import json
import pprint
import argparse
import importlib
import shutil

from tqdm import tqdm
import torch

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory

torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test CenterNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("-i", dest="image_path", help="image path",
                        default=None, type=str)
    parser.add_argument("-o", dest="result_path", help="result path",
                        default=None, type=str)

    args = parser.parse_args()
    return args


def demo(db, testiter, image_dir, result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)

    # 开始做inference
    demo_file = "test.{}".format(db.data)
    demoing = importlib.import_module(demo_file).demoing
    nnet.cuda()
    nnet.eval_mode()

    for image_file in tqdm(os.listdir(image_dir)):
        demoing(os.path.join(image_dir, image_file), db, nnet,
                os.path.join(result_dir, image_file))


if __name__ == "__main__":
    # 若当作脚本运行，使用parse_args；若在notebook中运行，使用Args类
    args = parse_args()
    # args = Args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    print("loading all datasets...")
    dataset = system_configs.dataset
    demo_db = datasets[dataset](configs["db"], demo=True)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(demo_db.configs)

    demo(demo_db, args.testiter, args.image_path, args.result_path)
