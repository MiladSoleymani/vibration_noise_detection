from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


import numpy as np

KNEIGHBORSPARAMGRID = {"n_neighbors": list(range(1, 31))}

KNEIGHBORSMODEL = KNeighborsClassifier()


LOGISTICREGRESSIONPARAMGRID = {
    "penalty": ["l1", "l2"],
    "C": np.logspace(-3, 3, 7),
    "solver": ["newton-cg", "lbfgs", "liblinear"],
}

LOGISTICREGRESSIONMODEL = LogisticRegression()

SVMPARAMGRID = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["linear", "poly", "rbf"],
}

SVMMODEL = svm.SVC()


DECISIONTREEClASSIFIERPARAMGRID = {
    "max_leaf_nodes": list(range(2, 100)),
    "min_samples_split": [2, 3, 4],
}

DECISIONTREEClASSIFIERMODEL = DecisionTreeClassifier()


RANDOMFORESTClASSIFIERPARAMGRID = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
}

RANDOMFORESTClASSIFIERMODEL = RandomForestClassifier(random_state=42)


XGBClASSIFIERPARAMGRID = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

XGBClASSIFIERMODEL = XGBClassifier(objective="binary:logistic", nthread=4, seed=42)
