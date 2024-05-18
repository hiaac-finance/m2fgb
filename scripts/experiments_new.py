import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import joblib
import datetime

import optuna
from optuna.samplers import RandomSampler

import os
import data
import models
import utils
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
import warnings

sys.stderr = open("stderr.txt", "w")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


SEED = 0


def get_model(model_name, random_state=None):
    """Helper function to get model class from model name."""
    if model_name == "M2FGB":

        def model(**params):
            return models.M2FGB(random_state=random_state, **params)

    elif model_name == "M2FGB_grad":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient_norm", random_state=random_state, **params
            )

    elif model_name == "M2FGB_onlyfair":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient_norm",
                fair_weight=1,
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_eod":

        def model(**params):
            return models.M2FGB(
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_grad_tpr":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient",
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_pr":

        def model(**params):
            return models.M2FGB(
                fairness_constraint="positive_rate",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_grad_pr":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient_norm",
                fairness_constraint="positive_rate",
                random_state=random_state,
                **params,
            )

    elif model_name == "LGBMClassifier":

        def model(**params):
            return models.LGBMClassifier(
                random_state=random_state, verbose=-1, **params
            )

    elif model_name == "FairGBMClassifier":

        def model(**params):
            return models.FairGBMClassifier(random_state=random_state, **params)

    elif model_name == "FairGBMClassifier_eod":

        def model(**params):
            return models.FairGBMClassifier(
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

    elif model_name == "FairClassifier_spd":

        def model(**params):
            return models.FairClassifier_Wrap(
                fairness_constraint="demographic_parity", **params
            )

    elif model_name == "MinMaxFair":

        def model(**params):
            return models.MinMaxFair(**params)

    elif model_name == "MinMaxFair_tpr":

        def model(**params):
            return models.MinMaxFair(fairness_constraint="tpr", **params)

    elif model_name == "MinimaxPareto":

        def model(**params):
            return models.MinimaxPareto(**params)

    return model


def get_param_spaces(model_name):
    """Helper function to get parameter space from model name."""
    if model_name not in [
        "M2FGB_eod",
        "M2FGB_pr",
        "M2FGB_grad_tpr",
        "M2FGB_grad_pr",
        "M2FGB_onlyfair",
        "FairGBMClassifier_eod",
        "FairClassifier_pr",
        "MinMaxFair_tpr",
    ]:
        return models.PARAM_SPACES[model_name]
    elif model_name == "M2FGB_eod" or model_name == "M2FGB_pr":
        return models.PARAM_SPACES["M2FGB"]
    elif model_name == "M2FGB_grad_tpr" or model_name == "M2FGB_grad_pr":
        return models.PARAM_SPACES["M2FGB_grad"]
    elif model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES["FairGBMClassifier"]
    elif model_name == "FairClassifier_spd":
        return models.PARAM_SPACES["FairClassifier"]
    elif model_name == "MinMaxFair_tpr":
        return models.PARAM_SPACES["MinMaxFair"]
    elif model_name == "M2FGB_onlyfair":
        param_space = models.PARAM_SPACES["M2FGB_grad"].copy()
        del param_space["fair_weight"]
        return param_space


def get_param_spaces_acsincome(model_name):
    """Helper function to get parameter space from model name."""
    if model_name not in [
        "M2FGB_eod",
        "M2FGB_pr",
        "M2FGB_grad_tpr",
        "M2FGB_grad_pr",
        "FairGBMClassifier_eod",
        "FairClassifier_spd",
        "MinMaxFair_tpr",       
    ]:
        return models.PARAM_SPACES_ACSINCOME[model_name]
    elif model_name == "M2FGB_tpr" or model_name == "M2FGB_pr":
        return models.PARAM_SPACES_ACSINCOME["M2FGB"]
    elif model_name == "M2FGB_grad_tpr" or model_name == "M2FGB_grad_pr":
        return models.PARAM_SPACES_ACSINCOME["M2FGB_grad"]
    elif model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES_ACSINCOME["FairGBMClassifier"]
    elif model_name == "FairClassifier_spd":
        return models.PARAM_SPACES_ACSINCOME["FairClassifier"]
    elif model_name == "MinMaxFair_tpr":
        return models.PARAM_SPACES_ACSINCOME["MinMaxFair"]


def get_subgroup_feature(dataset, X_train, X_val, X_test, n_groups=2):
    assert n_groups in [2, 4, 8]
    if n_groups == 2:
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
        elif dataset == "acsincome":
            A_train = X_train.SEX.astype(str)
            A_val = X_val.SEX.astype(str)
            A_test = X_test.SEX.astype(str)

    elif n_groups == 4:
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
                + (
                    (X_val.age_cat == "25 - 45") | (X_val.age_cat == "Less than 25")
                ).astype(str)
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
        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "1"
                elif race == "african_america":
                    return "2"
                elif race == "asian":
                    return "3"
                else:
                    return "4"

            A_train = X_train.RAC1P.apply(race_cat)
            A_val = X_val.RAC1P.apply(race_cat)
            A_test = X_test.RAC1P.apply(race_cat)

    elif n_groups == 8:
        if dataset == "german":

            def age_cat(age):
                if age < 30:
                    return "1"
                elif age < 40:
                    return "2"
                elif age < 50:
                    return "3"
                else:
                    return "4"

            A_train = (
                X_train.Gender.astype(str)
                + "_"
                + X_train.Age.apply(age_cat).astype(str)
            )
            A_val = (
                X_val.Gender.astype(str) + "_" + X_val.Age.apply(age_cat).astype(str)
            )
            A_test = (
                X_test.Gender.astype(str) + "_" + X_test.Age.apply(age_cat).astype(str)
            )
        elif dataset == "adult":

            def age_cat(age):
                if age < 30:
                    return "1"
                elif age < 40:
                    return "2"
                elif age < 50:
                    return "3"
                else:
                    return "4"

            A_train = (
                X_train.sex.astype(str) + "_" + X_train.age.apply(age_cat).astype(str)
            )
            A_val = X_val.sex.astype(str) + "_" + X_val.age.apply(age_cat).astype(str)
            A_test = (
                X_test.sex.astype(str) + "_" + X_test.age.apply(age_cat).astype(str)
            )
        elif dataset == "compas":

            def race_cat(race):
                if race == "African-American" or race == "Hispanic":
                    return "1"
                elif race == "Caucasian":
                    return "2"
                elif race == "Asian":
                    return "3"
                else:
                    return "4"

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
                + (
                    (X_val.age_cat == "25 - 45") | (X_val.age_cat == "Less than 25")
                ).astype(str)
            )
            A_test = (
                X_test.race.apply(race_cat)
                + "_"
                + (
                    (X_test.age_cat == "25 - 45") | (X_test.age_cat == "Less than 25")
                ).astype(str)
            )
        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "1"
                elif race == "african_america":
                    return "2"
                elif race == "asian":
                    return "3"
                else:
                    return "4"

            A_train = X_train.RAC1P.apply(race_cat) + "_" + X_train.SEX.astype(str)
            A_val = X_val.RAC1P.apply(race_cat) + "_" + X_val.SEX.astype(str)
            A_test = X_test.RAC1P.apply(race_cat) + "_" + X_test.SEX.astype(str)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    print(sensitive_map)
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def get_param_list(model_name, n_params):
    param_space = get_param_spaces(model_name)

    def sample_random_parameters(param_space):
        params = {}
        for key, value in param_space.items():
            if value["type"] == "int":
                if "log" in value and value["log"]:
                    params[key] = int(
                        np.exp(
                            np.random.uniform(
                                np.log(value["low"]),
                                np.log(value["high"]),
                            )
                        )
                    )
                else:
                    params[key] = np.random.randint(
                        value["low"],
                        value["high"],
                    )
            elif value["type"] == "float":
                if "log" in value and value["log"]:
                    params[key] = np.exp(
                        np.random.uniform(
                            np.log(value["low"]),
                            np.log(value["high"]),
                        )
                    )
                else:
                    params[key] = np.random.uniform(
                        value["low"],
                        value["high"],
                    )
            elif value["type"] == "str":
                params[key] = np.random.choice(value["options"])

        return params

    param_list = [sample_random_parameters(param_space) for _ in range(n_params)]
    return param_list


from functools import partial


def fit_model(model_name, params, X_train, Y_train, A_train):
    model_class = get_model(model_name, random_state=SEED)
    model = model_class(**params)
    model.fit(X_train, Y_train, A_train)
    return model


def train_models(
    model_name,
    param_list,
    X_train,
    Y_train,
    A_train,
    n_jobs,
):
    """Function to train a list of models based on a list of parameters"""
    if n_jobs > 1:
        pool = Pool(n_jobs)
        fit_model_partial = partial(
            fit_model,
            model_name,
            X_train=X_train,
            Y_train=Y_train.values,
            A_train=A_train.values,
        )
        model_list = list(
            tqdm(
                pool.imap_unordered(fit_model_partial, param_list),
                total=len(param_list),
            )
        )
        # model_list = pool.map(fit_model_partial, param_list)
        pool.close()
    else:
        model_list = []
        for params in param_list:
            model_class = get_model(model_name, SEED)
            model = model_class(**params)
            model.fit(X_train, Y_train, A_train)
    return model_list


def eval_model(
    alpha_list,
    fair_metric,
    model_list,
    thresh,
    X_val,
    Y_val,
    A_val,
    X_test,
    Y_test,
    A_test,
):
    """Evaluate model performance and fairness metrics."""
    scorer_list = [
        utils.get_combined_metrics_scorer(
            alpha=alpha,
            performance_metric="bal_acc",
            fairness_metric=fair_metric,
        )
        for alpha in alpha_list
    ]
    results_val = []
    results_test = []
    for m, model in tqdm(enumerate(model_list), total=len(model_list)):
        y_val_score = model.predict_proba(X_val)[:, 1]
        if thresh == "ks":
            thresh = utils.get_best_threshold(Y_val, y_val_score)
        else:
            thresh = 0.5

        y_val_pred = y_val_score > thresh
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = y_test_score > thresh

        bal_acc = balanced_accuracy_score(Y_val, y_val_pred)
        prec = precision_score(Y_val, y_val_pred)
        acc = accuracy_score(Y_val, y_val_pred)
        roc = roc_auc_score(Y_val, y_val_score)
        eq_loss = utils.equalized_loss_score(Y_val, y_val_score, A_val)
        eod = utils.equal_opportunity_score(Y_val, y_val_pred, A_val)
        spd = utils.statistical_parity_score(Y_val, y_val_pred, A_val)
        min_tpr = 1 - utils.min_true_positive_rate(Y_val, y_val_pred, A_val)
        min_pr = 1 - utils.min_positive_rate(Y_val, y_val_pred, A_val)
        min_bal_acc = 1 - utils.min_balanced_accuracy(Y_val, y_val_pred, A_val)

        for i, alpha in enumerate(alpha_list):
            score = scorer_list[i](Y_val, y_val_pred, A_val)

            results_val.append(
                {
                    "alpha": alpha,
                    "score": score,
                    "bal_acc": bal_acc,
                    "prec": prec,
                    "acc": acc,
                    "roc": roc,
                    "eq_loss": eq_loss,
                    "eod": eod,
                    "spd": spd,
                    "model": m,
                    "min_tpr": min_tpr,
                    "min_pr": min_pr,
                    "min_bal_acc": min_bal_acc,
                }
            )

        bal_acc = balanced_accuracy_score(Y_test, y_test_pred)
        acc = accuracy_score(Y_test, y_test_pred)
        roc = roc_auc_score(Y_test, y_test_score)
        eq_loss = utils.equalized_loss_score(Y_test, y_test_score, A_test)
        eod = utils.equal_opportunity_score(Y_test, y_test_pred, A_test)
        spd = utils.statistical_parity_score(Y_test, y_test_pred, A_test)
        min_tpr = 1 - utils.min_true_positive_rate(Y_test, y_test_pred, A_test)
        min_pr = 1 - utils.min_positive_rate(Y_test, y_test_pred, A_test)
        min_bal_acc = 1 - utils.min_balanced_accuracy(Y_test, y_test_pred, A_test)

        for i, alpha in enumerate(alpha_list):
            score = scorer_list[i](Y_test, y_test_pred, A_test)

            results_test.append(
                {
                    "alpha": alpha,
                    "score": score,
                    "bal_acc": bal_acc,
                    "prec": prec,
                    "acc": acc,
                    "roc": roc,
                    "eq_loss": eq_loss,
                    "eod": eod,
                    "spd": spd,
                    "model": m,
                    "min_tpr": min_tpr,
                    "min_pr": min_pr,
                    "min_bal_acc": min_bal_acc,
                }
            )

    results_val = pd.DataFrame(results_val)
    results_test = pd.DataFrame(results_test)
    return results_val, results_test


def run_trial(trial, X_train, Y_train, A_train, model_class, param_space, model_list):
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

    print(params)
    model = model_class(**params)
    model.fit(X_train, Y_train, A_train)
    model_list.append(model)
    return 0.5


def run_subgroup_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))

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

    for i in tqdm(range(args["n_folds"])):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, args["n_folds"], SEED
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_subgroup_feature(
            args["dataset"], X_train, X_val, X_test, args["n_groups"]
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

        if args["dataset"] != "acsincome":
            param_space = get_param_spaces(args["model_name"])
        else:
            param_space = get_param_spaces_acsincome(args["model_name"])

        study = optuna.create_study(
            direction="maximize", sampler=RandomSampler(seed=SEED)
        )
        model_list = []
        objective = lambda trial: run_trial(
            trial,
            X_train,
            Y_train,
            A_train,
            get_model(args["model_name"], random_state=SEED),
            param_space,
            model_list,
        )
        study.optimize(
            objective,
            n_trials=args["n_params"],
            n_jobs=args["n_jobs"],
            show_progress_bar=True,
        )
        trials_df = study.trials_dataframe(attrs=("number", "duration", "params"))
        trials_df.to_csv(
            os.path.join(args["output_dir"], f"trials_fold_{i}.csv"), index=False
        )

        results_val, results_test = eval_model(
            args["alpha_list"],
            args["fair_metric"],
            model_list,
            args["thresh"],
            X_val,
            Y_val,
            A_val,
            X_test,
            Y_test,
            A_test,
        )

        # save results of fold
        results_val.to_csv(
            os.path.join(args["output_dir"], f"validation_fold_{i}.csv"), index=False
        )
        results_test.to_csv(
            os.path.join(args["output_dir"], f"test_fold_{i}.csv"), index=False
        )


def experiment1():
    n_folds = 10
    thresh = "ks"
    alpha_list = [i / 20 for i in range(0, 21)]
    n_jobs = 10
    fair_metric = "min_bal_acc"

    datasets = [
        # "german",
        # "compas",
        # "adult",
        "acsincome"
    ]
    n_groups_list = [2, 4, 8]
    model_name_list = [
        # "M2FGB_grad",
        # "FairGBMClassifier",
        # "MinMaxFair",
        # "LGBMClassifier",
        # "FairClassifier"
        # "M2FGB",
        "MinimaxPareto"
    ]

    n_params = 500
    for dataset in datasets:
        if dataset == "acsincome":
            n_params = 250
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if dataset == "acsincome":
                        n_params = 25
                    else:
                        n_params = 100

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = (
                    f"../results/experiment_{n_groups}_groups/{dataset}/{model_name}"
                )
                args = {
                    "dataset": dataset,
                    "alpha_list": alpha_list,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_folds": n_folds,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "fair_metric": fair_metric,
                    "n_jobs": n_jobs,
                    "thresh": thresh,
                }
                run_subgroup_experiment(args)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment2():
    n_folds = 10
    thresh = "ks"
    alpha_list = [i / 20 for i in range(0, 21)]
    n_jobs = 10
    fair_metric = "min_pr"

    datasets = [
        "german",
        "compas",
        "acsincome",
    ]
    n_groups_list = [
        # 2,
        4,
        8,
    ]
    model_name_list = [
        "M2FGB_grad_pr",
        # "FairGBMClassifier",
        # "MinMaxFair",
        "LGBMClassifier",
        # "FairClassifier"
        # "M2FGB",
        # "MinimaxPareto"
    ]

    n_params = 100
    for dataset in datasets:
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if dataset == "acsincome":
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = (
                    f"../results/experiment_{n_groups}_pr/{dataset}/{model_name}"
                )
                args = {
                    "dataset": dataset,
                    "alpha_list": alpha_list,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_folds": n_folds,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "fair_metric": fair_metric,
                    "n_jobs": n_jobs,
                    "thresh": thresh,
                }
                run_subgroup_experiment(args)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment3():
    n_folds = 10
    thresh = "ks"
    alpha_list = [i / 20 for i in range(0, 21)]
    n_jobs = 10
    fair_metric = "min_tpr"

    datasets = [
        #"german",
        #"compas",
        "acsincome",
    ]
    n_groups_list = [
        # 2,
        # 4,
        8,
    ]
    model_name_list = [
        #"MinMaxFair_tpr",
        "LGBMClassifier",
        "M2FGB_grad_tpr",
        
    ]

    n_params = 100
    for dataset in datasets:
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair_tpr" or model_name == "MinimaxPareto":
                    if dataset == "acsincome":
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = (
                    f"../results/experiment_{n_groups}_tpr/{dataset}/{model_name}"
                )
                args = {
                    "dataset": dataset,
                    "alpha_list": alpha_list,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_folds": n_folds,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "fair_metric": fair_metric,
                    "n_jobs": n_jobs,
                    "thresh": thresh,
                }
                run_subgroup_experiment(args)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def main():
    import lightgbm as lgb
    import fairgbm

    lgb.register_logger(utils.CustomLogger())
    fairgbm.register_logger(utils.CustomLogger())

    experiment3()


if __name__ == "__main__":
    main()
