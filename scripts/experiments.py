import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import logging


class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger("lightgbm_custom")
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        # Suppress warnings by not doing anything
        pass

    def error(self, message):
        self.logger.error(message)


import lightgbm as lgb

lgb.register_logger(CustomLogger())

import sys
import os
import glob
import data
import models
import utils
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from fairgbm import FairGBMClassifier
from lightgbm import LGBMClassifier


SEED = 0


def run_trial(
    trial,
    scorer,
    X_train,
    Y_train,
    A_train,
    X_val,
    Y_val,
    A_val,
    model_class,
    param_space,
):
    """Function to run a single trial of optuna."""
    params = {}
    for name, values in param_space.items():
        if values["type"] == "int":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_int(name, **values_cp)
        elif values["type"] == "categorical":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_categorical(name, **values_cp)
        elif values["type"] == "float":  # corrected this line
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_float(name, **values_cp)

    model = model_class(**params)
    if isinstance(model, FairGBMClassifier):
        model.fit(X_train, Y_train, constraint_group=A_train)
    elif isinstance(model, LGBMClassifier):
        model.fit(X_train, Y_train)
    else:
        model.fit(X_train, Y_train, A_train)

    Y_val_score = model.predict_proba(X_val)[:, 1]
    thresh = utils.get_best_threshold(Y_val, Y_val_score)
    Y_val_pred = Y_val_score > thresh
    return scorer(Y_val, Y_val_pred, A_val)


def get_model(model_name, random_state=None):
    """Helper function to get model class from model name."""
    if model_name == "XtremeFair":

        def model(**params):
            return models.XtremeFair(random_state=random_state, **params)

    elif model_name == "XtremeFair_grad":

        def model(**params):
            return models.XtremeFair(
                dual_learning="gradient", random_state=random_state, **params
            )

    elif model_name == "MMBFair":

        def model(**params):
            return models.MMBFair(random_state=random_state, **params)

    elif model_name == "MMBFair_grad":

        def model(**params):
            return models.MMBFair(
                dual_learning="gradient", random_state=random_state, **params
            )

    elif model_name == "MMBFair_eod":

        def model(**params):
            return models.MMBFair(
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "MMBFair_grad_eod":

        def model(**params):
            return models.MMBFair(
                dual_learning="gradient",
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "LGBMClassifier":

        def model(**params):
            return LGBMClassifier(random_state=random_state, verbose=-1, **params)

    elif model_name == "FairGBMClassifier":

        def model(**params):
            return FairGBMClassifier(random_state=random_state, **params)

    elif model_name == "FairGBMClassifier_eod":

        def model(**params):
            return FairGBMClassifier(
                constraint_type="FNR", random_state=random_state, **params
            )

    elif model_name == "ExponentiatedGradient":

        def model(**params):
            return models.ExponentiatedGradient_Wrap(
                random_state=random_state, **params
            )

    elif model_name == "FairClassifier":

        def model(**params):
            return models.FairClassifier_Wrap(**params)

    return model


def get_param_spaces(model_name):
    """Helper function to get parameter space from model name."""
    if model_name == "XtremeFair":
        return models.PARAM_SPACES["XtremeFair"]
    elif model_name == "XtremeFair_grad":
        return models.PARAM_SPACES["XtremeFair_grad"]
    elif model_name == "MMBFair" or model_name == "MMBFair_eod":
        return models.PARAM_SPACES["MMBFair"]
    elif model_name == "MMBFair_grad" or model_name == "MMBFair_grad_eod":
        return models.PARAM_SPACES["MMBFair_grad"]
    elif model_name == "LGBMClassifier":
        return models.PARAM_SPACES["LGBMClassifier"]
    elif model_name == "FairGBMClassifier" or model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES["FairGBMClassifier"]
    elif model_name == "ExponentiatedGradient":
        return models.PARAM_SPACES["ExponentiatedGradient"]
    elif model_name == "FairClassifier":
        return models.PARAM_SPACES["FairClassifier"]


def get_group_feature(dataset, X_train, X_val, X_test):
    """Function to get the sensitive attribute that defines binary group."""
    if dataset == "german":
        A_train = X_train.Gender.astype(str)
        A_val = X_val.Gender.astype(str)
        A_test = X_test.Gender.astype(str)
    elif dataset == "adult":
        A_train = X_train.sex.astype(str)
        A_val = X_val.sex.astype(str)
        A_test = X_test.sex.astype(str)
    elif dataset == "compas":
        A_train = X_train.race == "Caucasian"
        A_val = X_val.race == "Caucasian"
        A_test = X_test.race == "Caucasian"

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def get_subgroup1_feature(dataset, X_train, X_val, X_test):
    """Function to get the sensitive attribute that defines 4 groups."""
    if dataset == "german":
        A_train = X_train.Gender.astype(str) + "_" + (X_train.Age > 50).astype(str)
        A_val = X_val.Gender.astype(str) + "_" + (X_val.Age > 50).astype(str)
        A_test = X_test.Gender.astype(str) + "_" + (X_test.Age > 50).astype(str)
    elif dataset == "compas":
        A_train = (
            (X_train.race == "Caucasian").astype(str)
            + "_"
            + (
                (X_train.age_cat == "25 - 45") | (X_train.age_cat == "Less than 25")
            ).astype(str)
        )
        A_val = (
            (X_val.race == "Caucasian").astype(str)
            + "_"
            + ((X_val.age_cat == "25 - 45") | (X_val.age_cat == "Less than 25")).astype(
                str
            )
        )
        A_test = (
            (X_test.race == "Caucasian").astype(str)
            + "_"
            + (
                (X_test.age_cat == "25 - 45") | (X_test.age_cat == "Less than 25")
            ).astype(str)
        )
    elif dataset == "adult":
        A_train = X_train.sex.astype(str) + "_" + (X_train.age > 50).astype(str)
        A_val = X_val.sex.astype(str) + "_" + (X_val.age > 50).astype(str)
        A_test = X_test.sex.astype(str) + "_" + (X_test.age > 50).astype(str)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def get_subgroup2_feature(dataset, X_train, X_val, X_test):
    """Function to get the sensitive attribute that defines 8 groups."""

    def age_cat(age):
        if age < 30:
            return "1"
        elif age < 40:
            return "2"
        elif age < 50:
            return "3"
        else:
            return "4"

    def race_cat(race):
        if race == "African-American" or race == "Hispanic":
            return "1"
        elif race == "Caucasian":
            return "2"
        elif race == "Asian":
            return "3"
        else:
            return "4"

    if dataset == "german":
        A_train = (
            X_train.Gender.astype(str) + "_" + X_train.Age.apply(age_cat).astype(str)
        )
        A_val = X_val.Gender.astype(str) + "_" + X_val.Age.apply(age_cat).astype(str)
        A_test = X_test.Gender.astype(str) + "_" + X_test.Age.apply(age_cat).astype(str)
    elif dataset == "adult":
        A_train = X_train.sex.astype(str) + "_" + X_train.age.apply(age_cat).astype(str)
        A_val = X_val.sex.astype(str) + "_" + X_val.age.apply(age_cat).astype(str)
        A_test = X_test.sex.astype(str) + "_" + X_test.age.apply(age_cat).astype(str)
    elif dataset == "compas":
        A_train = (
            X_train.race.apply(race_cat)
            + "_"
            + (
                (X_train.age_cat == "25 - 45") | (X_train.age_cat == "Less than 25")
            ).astype(str)
        )
        A_val = (
            X_val.race.apply(race_cat)
            + "_"
            + ((X_val.age_cat == "25 - 45") | (X_val.age_cat == "Less than 25")).astype(
                str
            )
        )
        A_test = (
            X_test.race.apply(race_cat)
            + "_"
            + (
                (X_test.age_cat == "25 - 45") | (X_test.age_cat == "Less than 25")
            ).astype(str)
        )

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def eval_model(y_true, y_score, y_pred, A):
    """Evaluate model performance and fairness metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_score)
    eq_loss = utils.equalized_loss_score(y_true, y_score, A)
    eod = utils.equal_opportunity_score(y_true, y_pred, A)
    spd = utils.statistical_parity_score(y_true, y_pred, A)
    metrics = {
        "acc": acc,
        "bal_acc": bal_acc,
        "roc": roc,
        "eq_loss": eq_loss,
        "eod": eod,
        "spd": spd,
    }

    pr_list = []
    tpr_list = []
    bal_acc_list = []

    for g in np.unique(A):
        bool_g = A == g
        pr_list.append(y_pred[bool_g].mean())
        tpr_list.append(y_pred[bool_g & (y_true == 1)].mean())
        bal_acc_list.append(balanced_accuracy_score(y_true[bool_g], y_pred[bool_g]))

    metrics["tpr_min"] = min(tpr_list)
    metrics["tpr_max"] = max(tpr_list)
    metrics["tpr_mean"] = np.mean(tpr_list)
    metrics["pr_min"] = min(pr_list)
    metrics["pr_max"] = max(pr_list)
    metrics["pr_mean"] = np.mean(pr_list)
    metrics["bal_acc_min"] = min(bal_acc_list)
    metrics["bal_acc_max"] = max(bal_acc_list)
    metrics["bal_acc_mean"] = np.mean(bal_acc_list)

    return metrics


def run_group_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))
    results = []

    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), data.NUM_FEATURES[args["dataset"]]),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                data.CAT_FEATURES[args["dataset"]],
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")

    scorer = utils.get_combined_metrics_scorer(
        alpha=args["alpha"], performance_metric="bal_acc", fairness_metric="eod"
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, SEED
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_group_feature(
            args["dataset"], X_train, X_val, X_test
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

        model_class = get_model(args["model_name"], random_state=SEED)
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: run_trial(
            trial,
            scorer,
            X_train,
            Y_train,
            A_train,
            X_val,
            Y_val,
            A_val,
            model_class,
            get_param_spaces(args["model_name"]),
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=11)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        if isinstance(model, FairGBMClassifier):
            model.fit(X_train, Y_train, constraint_group=A_train)
        elif isinstance(model, LGBMClassifier):
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, Y_train, A_train)

        y_val_score = model.predict_proba(X_val)[:, 1]
        thresh = utils.get_best_threshold(Y_val, y_val_score)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = y_test_score > thresh
        metrics = eval_model(Y_test, y_test_score, y_test_pred, A_test)

        best_params["threshold"] = thresh
        # save results of fold
        joblib.dump(model, os.path.join(args["output_dir"], f"model_{i}.pkl"))
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")
        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args["output_dir"], "results.csv"), index=False)


def run_subgroup_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))
    results = []

    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), data.NUM_FEATURES[args["dataset"]]),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                data.CAT_FEATURES[args["dataset"]],
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")

    scorer = utils.get_combined_metrics_scorer(
        alpha=args["alpha"], performance_metric="bal_acc", fairness_metric="eod"
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, SEED
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_subgroup1_feature(
            args["dataset"], X_train, X_val, X_test
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

        model_class = get_model(args["model_name"], random_state=SEED)
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: run_trial(
            trial,
            scorer,
            X_train,
            Y_train,
            A_train,
            X_val,
            Y_val,
            A_val,
            model_class,
            get_param_spaces(args["model_name"]),
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=11)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        if isinstance(model, FairGBMClassifier):
            model.fit(X_train, Y_train, constraint_group=A_train)
        elif isinstance(model, LGBMClassifier):
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, Y_train, A_train)
        y_val_score = model.predict_proba(X_val)[:, 1]
        thresh = utils.get_best_threshold(Y_val, y_val_score)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = y_test_score > thresh
        metrics = eval_model(Y_test, y_test_score, y_test_pred, A_test)
        best_params["threshold"] = thresh
        joblib.dump(model, os.path.join(args["output_dir"], f"model_{i}.pkl"))
        # save best params
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")
        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args["output_dir"], "results.csv"), index=False)


def run_fairness_goal_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))
    results = []

    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), data.NUM_FEATURES[args["dataset"]]),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                data.CAT_FEATURES[args["dataset"]],
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")

    scorer = utils.get_fairness_goal_scorer(
        fairness_goal=args["goal"], performance_metric="bal_acc", fairness_metric="eod"
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, SEED
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_group_feature(
            args["dataset"], X_train, X_val, X_test
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

        model_class = get_model(args["model_name"], random_state=SEED)
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: run_trial(
            trial,
            scorer,
            X_train,
            Y_train,
            A_train,
            X_val,
            Y_val,
            A_val,
            model_class,
            get_param_spaces(args["model_name"]),
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=7)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        if isinstance(model, FairGBMClassifier):
            model.fit(X_train, Y_train, constraint_group=A_train)
        elif isinstance(model, LGBMClassifier):
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, Y_train, A_train)
        y_val_score = model.predict_proba(X_val)[:, 1]
        thresh = utils.get_best_threshold(Y_val, y_val_score)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = y_test_score > thresh
        metrics = eval_model(Y_test, y_test_score, y_test_pred, A_test)
        best_params["threshold"] = thresh
        joblib.dump(model, os.path.join(args["output_dir"], f"model_{i}.pkl"))
        # save best params
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")
        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args["output_dir"], "results.csv"))


def run_subgroup2_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))
    results = []

    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), data.NUM_FEATURES[args["dataset"]]),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                data.CAT_FEATURES[args["dataset"]],
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")

    scorer = utils.get_combined_metrics_scorer(
        alpha=args["alpha"], performance_metric="bal_acc", fairness_metric="eod"
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, SEED
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_subgroup2_feature(
            args["dataset"], X_train, X_val, X_test
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

        model_class = get_model(args["model_name"], random_state=SEED)
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: run_trial(
            trial,
            scorer,
            X_train,
            Y_train,
            A_train,
            X_val,
            Y_val,
            A_val,
            model_class,
            get_param_spaces(args["model_name"]),
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=11)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        if isinstance(model, FairGBMClassifier):
            model.fit(X_train, Y_train, constraint_group=A_train)
        elif isinstance(model, LGBMClassifier):
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, Y_train, A_train)
        y_val_score = model.predict_proba(X_val)[:, 1]
        thresh = utils.get_best_threshold(Y_val, y_val_score)
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = y_test_score > thresh
        metrics = eval_model(Y_test, y_test_score, y_test_pred, A_test)
        best_params["threshold"] = thresh
        joblib.dump(model, os.path.join(args["output_dir"], f"model_{i}.pkl"))
        # save best params
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")
        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args["output_dir"], "results.csv"), index=False)


def experiment1():
    datasets = ["german", "compas", "adult"]
    model_names = [
        # "LGBMClassifier",
        # "FairGBMClassifier",
        # "ExponentiatedGradient",  # TODO
        # "FairClassifier",
        # "MMBFair",
        # "MMBFair_grad",
        "MMBFair_eod",
        "MMBFair_grad_eod",
    ]
    alphas = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alphas = [0.7, 0.9, 0.2]

    for dataset in datasets:
        for alpha in alphas:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "output_dir": f"../results/group_experiment/{dataset}/{model_name}_{alpha}",
                    "model_name": model_name,
                    "n_trials": 50,
                }
                print(f"{dataset} {model_name} {alpha}")
                run_group_experiment(args)


def experiment2():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairGBMClassifier_eod",
        "MMBFair",
        "MMBFair_grad",
        "MMBFair_eod",
        "MMBFair_grad_eod",
    ]
    alphas = [0.75]
    for dataset in datasets:
        for alpha in alphas:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "output_dir": f"../results/subgroup_experiment/{dataset}/{model_name}_{alpha}",
                    "model_name": model_name,
                    "n_trials": 100,
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup_experiment(args)


def experiment3():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairGBMClassifier_eod",
        "MMBFair",
        "MMBFair_grad",
        "MMBFair_eod",
        "MMBFair_grad_eod",
    ]
    alphas = [0.75]
    for dataset in datasets:
        for alpha in alphas:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "output_dir": f"../results/subgroup2_experiment/{dataset}/{model_name}_{alpha}",
                    "model_name": model_name,
                    "n_trials": 100,
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup2_experiment(args)


def experiment3():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "XtremeFair",
        "XtremeFair_grad",
        # "ExponentiatedGradient",  # TODO
        "FairClassifier",
    ]
    goals = [0.95]
    for dataset in datasets:
        for goal in goals:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "goal": goal,
                    "output_dir": f"../results/fairness_goal_experiment/{dataset}/{model_name}_{goal}",
                    "model_name": model_name,
                    "n_trials": 100,
                }
                print(f"{dataset} {model_name} {goal}")
                run_fairness_goal_experiment(args)


def main():
    # experiment1()  # (binary groups)

    experiment2()  # (4 groups)

    experiment3()  # (8 groups)

    # experiment3() # (fairness goal)

    # experiment 5 (EOD)

    # experiment 6 (SPD)


if __name__ == "__main__":
    main()
