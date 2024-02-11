import sys
import os

sys.path.append(os.getcwd())

from typing import Dict
import numpy as np
import pandas as pd

from utils.utils import (
    select_model,
    feature_extraction,
    concate_sensor_features,
)

from data.transforms import ToNumpy, Compose

from data.dataset import VibrationNoiseDataset

import argparse

import warnings
import json


def predict(conf: Dict) -> None:

    warnings.filterwarnings("ignore")

    cc_train_data = VibrationNoiseDataset(
        conf["train_data_path"],
        conf["time_points"],
        conf["cutoff"],
        conf["transforms"],
        "train",
    )

    train_data = cc_train_data.__getitem__()

    train_data = feature_extraction([conf["feature_name"]], train_data)
    train_data = pd.DataFrame(train_data)

    x_train, y_train, meta_data_train = concate_sensor_features(
        train_data, conf["feature_name"]
    )

    cc_test_data = VibrationNoiseDataset(
        conf["test_data_path"],
        conf["time_points"],
        conf["cutoff"],
        conf["transforms"],
        "test",
    )

    test_data = cc_test_data.__getitem__()

    test_data = feature_extraction([conf["feature_name"]], test_data)
    test_data = pd.DataFrame(test_data)

    x_test, _, meta_data_test = concate_sensor_features(test_data, conf["feature_name"])

    with open(
        conf["model_path"],
    ) as json_file:
        model_params = json.load(json_file)

    model_params["model_params"]["probability"] = True

    clf = select_model({"grid_search": "OFF", "model": model_params["model_name"]})
    clf.set_params(**model_params["model_params"])  # **clf.best_params_
    clf.fit(x_train, y_train)

    print("Probability is based on percentage\n")
    for engine_name in meta_data_test["engine_names"].drop_duplicates():
        predicts = clf.predict(x_test)
        probaility = clf.predict_proba(x_test)

        print(f"Predicting {engine_name}...")

        for idx, _ in enumerate(probaility):
            print(
                f"part_number {idx + 1}:\n"
                f"   OK probability={ _[0]:.2f}    NG probability={_[1]:.2f}\n"
            )

        predict = (
            1
            if np.count_nonzero(predicts == 1) > np.count_nonzero(predicts == 0)
            else 0
        )

        print(f"{engine_name} is OK") if predict == 0 else print(f"{engine_name} is NG")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="../vibration_noise_detection/datasets/vibration_data",
    )

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="../vibration_noise_detection/datasets/test_data",
    )

    parser.add_argument("--cutoff", type=int, default=10)

    parser.add_argument(
        "--model_path",
        type=str,
        default="../vibration_noise_detection/best_model_params/svm_psd.json",
    )

    parser.add_argument(
        "--feature_name",
        type=str,
        default="psd",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)

    time_points = [0, 131710, 214029, 274396, 321592, 360007, 398422, -1]
    # time_points = [214029, 274396, 321592, 360007]
    # time_points = [214029, 274396, 321592, 360007, 398422, -1]
    # time_points = [0, 131710, 214029, 274396, 321592]
    # time_points = [0, 10000, 20000, 30000, -1]
    transforms = Compose([ToNumpy()])

    temp_conf = conf.copy()
    temp_conf["time_points"] = time_points
    temp_conf["transforms"] = transforms

    predict(temp_conf)
