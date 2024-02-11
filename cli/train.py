import sys
import os

sys.path.append(os.getcwd())

from typing import Dict
import pandas as pd

from utils.utils import (
    select_model,
    fit_eval,
    voting,
    feature_extraction,
    concate_sensor_features,
)

from data.transforms import ToNumpy, Compose

from data.dataset import VibrationNoiseDataset

import argparse
import itertools

import json

import wandb


def train(conf: Dict) -> None:

    cc = VibrationNoiseDataset(
        conf["data_path"],
        conf["time_points"],
        conf["cutoff"],
        conf["transforms"],
        "train",
    )

    data = cc.__getitem__()

    data = feature_extraction([conf["feature_name"]], data)
    data = pd.DataFrame(data)

    x, y, meta_data = concate_sensor_features(data, conf["feature_name"])

    wandb_run = wandb.init(project=conf["proj"], group=conf["wb_group"], reinit=True)
    wandb.config.update(conf)

    clf = select_model({"grid_search": "ON", "model": conf["model"], "cv": conf["cv"]})

    best_params = fit_eval(clf, x, y, conf["cv"])

    voting(
        {"grid_search": "OFF", "model": conf["model"]},
        clf.best_params_,
        x,
        y,
        meta_data,
    )

    with open(
        os.path.join(conf["out_file"], f'{conf["model"]}_{conf["feature_name"]}.json'),
        "w",
    ) as json_file:

        model_params = {"model_name": conf["model"], "model_params": best_params}
        json.dump(model_params, json_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, default="test_project")
    parser.add_argument("--wb_group", type=str, default="test")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../vibration_noise_detection/datasets/vibration_data",
    )
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--cv", type=int, default=5)

    parser.add_argument(
        "--out_file", type=str, default="../vibration_noise_detection/best_model_params"
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    configs = {
        "feature_names": [
            {"feature_name": "psd"},
        ],
        "models": [
            {"model": "KNeighborsClassifier"},
            {"model": "svm"},
            {"model": "DecisionTreeClassifier"},
            {"model": "XGBClassifier"},
        ],
    }

    time_points = [0, 131710, 214029, 274396, 321592, 360007, 398422, -1]
    # time_points = [214029, 274396, 321592, 360007]
    # time_points = [214029, 274396, 321592, 360007, 398422, -1]
    # time_points = [0, 131710, 214029, 274396, 321592]
    # time_points = [0, 10000, 20000, 30000, -1]
    # time_points = [0, -1]

    transforms = Compose([ToNumpy()])

    for model, feature in itertools.product(
        configs["models"], configs["feature_names"]
    ):
        temp_conf = conf.copy()
        temp_conf.update(model)
        temp_conf.update(feature)
        temp_conf["time_points"] = time_points
        temp_conf["transforms"] = transforms
        temp_conf["wb_group"] = f"{model['model']}_{feature['feature_name']}"

        train(temp_conf)
