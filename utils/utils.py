from models.ml_models import *
from data.signal_features import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


import numpy as np
import pandas as pd

import warnings
from typing import (
    Dict,
    List,
    Tuple,
)

import wandb


def select_model(
    opts,
):  ## in train.py  if grid_search on use if clf.get_best_estimator
    if opts["grid_search"] == "ON":

        if opts["model"] == "KNeighborsClassifier":
            clf = GridSearchCV(
                estimator=KNEIGHBORSMODEL,
                param_grid=KNEIGHBORSPARAMGRID,
                cv=opts["cv"],
                verbose=1,
            )

        elif opts["model"] == "LogisticRegression":
            clf = GridSearchCV(
                estimator=LOGISTICREGRESSIONMODEL,
                param_grid=LOGISTICREGRESSIONPARAMGRID,
                cv=opts["cv"],
                verbose=1,
            )

        elif opts["model"] == "svm":
            clf = GridSearchCV(
                estimator=SVMMODEL, param_grid=SVMPARAMGRID, cv=opts["cv"], verbose=1
            )

        elif opts["model"] == "DecisionTreeClassifier":
            clf = GridSearchCV(
                estimator=DECISIONTREEClASSIFIERMODEL,
                param_grid=DECISIONTREEClASSIFIERPARAMGRID,
                cv=opts["cv"],
                verbose=1,
            )

        elif opts["model"] == "RandomForestClassifier":
            clf = GridSearchCV(
                estimator=RANDOMFORESTClASSIFIERMODEL,
                param_grid=RANDOMFORESTClASSIFIERPARAMGRID,
                cv=opts["cv"],
                verbose=1,
            )

        elif opts["model"] == "XGBClassifier":
            clf = GridSearchCV(
                estimator=XGBClASSIFIERMODEL,
                param_grid=XGBClASSIFIERPARAMGRID,
                cv=opts["cv"],
                verbose=1,
            )

        else:
            print(f"unknown model type {opts['model']}")
            clf = None

    elif opts["grid_search"] == "OFF":

        if opts["model"] == "KNeighborsClassifier":
            clf = KNEIGHBORSMODEL

        elif opts["model"] == "LogisticRegression":
            clf = LOGISTICREGRESSIONMODEL

        elif opts["model"] == "svm":
            clf = SVMMODEL

        elif opts["model"] == "DecisionTreeClassifier":
            clf = DECISIONTREEClASSIFIERMODEL

        elif opts["model"] == "RandomForestClassifier":
            clf = RANDOMFORESTClASSIFIERMODEL

        elif opts["model"] == "XGBClassifier":
            clf = XGBClASSIFIERMODEL

        else:
            print(f"unknown model type {opts['model']}")
            clf = None

    return clf


def feature_extraction(feature_names: List[str], data: Dict) -> Dict:

    for feature_name in feature_names:

        if feature_name == "psd":

            features = []
            for _ in data["signals"]:
                features.append(psd(_, fs=1000))

            data[feature_name] = features

        elif feature_name == "static_features":

            features = []
            for _ in data["signals"]:
                features.append(static_features(np.array([_])))

            data[feature_name] = features

        else:
            print(f"unknown feature_name type {feature_name}")

    return data


def concate_sensor_features(
    data: Dict, feature_name: str
) -> Tuple[np.array, np.array, pd.DataFrame]:

    sensor_1 = (
        data.query("`sensor_number` == 1")
        .sort_values(by=["engine_names"])
        .sort_values(by=["part_numbers"])
    )
    sensor_2 = (
        data.query("`sensor_number` == 2")
        .sort_values(by=["engine_names"])
        .sort_values(by=["part_numbers"])
    )
    meta_data = sensor_1[["engine_names", "part_numbers", "labels"]]

    y = np.array(sensor_1["labels"])

    sensor_1 = np.stack(sensor_1[feature_name], axis=0).squeeze()
    sensor_2 = np.stack(sensor_2[feature_name], axis=0).squeeze()

    x = np.concatenate((sensor_1, sensor_2), axis=1)

    return x, y, meta_data


def fit_eval(clf, x: np.array, y: np.array, cv: int) -> Dict:

    print("*** GridSearch on Model...")
    print(f"*** Model name : {clf.estimator.__class__}")  # .__name__

    warnings.filterwarnings("ignore")

    clf.fit(x, y)
    print("Done.")

    print("Prediction on whole data")
    preds = clf.best_estimator_.predict(x)
    print("Done.")
    print(
        classification_report(
            y,
            preds,
            labels=[i for i in range(2)],
            target_names=["OK", "NG"],
            zero_division=0,
        )
    )

    scores = cross_val_score(clf.best_estimator_, x, y, cv=cv)
    print(
        f"Accuracy score via cross-validation:\n"
        f"{scores.mean():.3f} ± {scores.std():.3f}"
    )

    temp_log = {}

    for _ in range(cv):
        temp_log[f"val_score_{_}"] = scores[_]

    temp_log[f"mean cross_val_score"] = scores.mean()
    temp_log[f"std cross_val_score"] = scores.std()

    if len(temp_log) > 0:
        wandb.log(temp_log)

    return clf.best_params_


def voting(
    opts: Dict, best_params: Dict, x: np.array, y: np.array, meta_data: pd.DataFrame
):

    prediction = {"engine_names": [], "prediction": [], "true_lable": []}

    for engine_name in meta_data["engine_names"].drop_duplicates():

        train_idx = np.array(meta_data["engine_names"]) != engine_name
        test_idx = np.array(meta_data["engine_names"]) == engine_name

        clf = select_model(opts)
        clf.set_params(**best_params)  # **clf.best_params_
        clf.fit(x[train_idx], y[train_idx])

        predicts = clf.predict(x[test_idx])

        prediction["engine_names"].append(engine_name)
        prediction["true_lable"].append(y[test_idx][0])
        prediction["prediction"].append(
            1
            if np.count_nonzero(predicts == 1) > np.count_nonzero(predicts == 0)
            else 0
        )

    voting_accuracy = accuracy_score(prediction["true_lable"], prediction["prediction"])

    print(f"Accuracy score via voting:\n" f"{voting_accuracy:.3f}")
    print("\n")

    temp_log = {}
    temp_log[f"voting_accuracy"] = voting_accuracy

    if len(temp_log) > 0:
        wandb.log(temp_log)
