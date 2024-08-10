import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import joblib
import datetime
import pickle as pkl

import optuna
from optuna.samplers import RandomSampler

import os
import data
import models
import utils
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    log_loss,
    recall_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


SEED = 0
np.random.seed(SEED)


def get_model(model_name, random_state=None):
    """Helper function to get model class from model name."""
    if model_name == "M2FGBClassifier":

        def model(**params):
            return models.M2FGBClassifier(random_state=random_state, **params)
        
    elif model_name == "M2FGBClassifier_v1":

        def model(**params):
            return models.M2FGBClassifier(dual_learning="gradient_norm", random_state=random_state, **params)

    elif model_name == "M2FGBClassifier_tpr":

        def model(**params):
            return models.M2FGBClassifier(
                fairness_constraint="true_positive_rate",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGBClassifier_pr":

        def model(**params):
            return models.M2FGBClassifier(
                fairness_constraint="positive_rate",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGBRegressor":

        def model(**params):
            return models.M2FGBRegressor(random_state=random_state, **params)

    elif model_name == "LGBMClassifier":

        def model(**params):
            return models.LGBMClassifier(random_state=random_state, **params)

    elif model_name == "LGBMRegressor":

        def model(**params):
            return models.LGBMRegressor(random_state=random_state, **params)

    elif model_name == "FairGBMClassifier":

        def model(**params):
            return models.FairGBMClassifier(random_state=random_state, **params)

    elif model_name == "FairGBMClassifier_eod":

        def model(**params):
            return models.FairGBMClassifier(
                constraint_type="FNR", random_state=random_state, **params
            )

    elif model_name == "MinMaxFair":

        def model(**params):
            return models.MinMaxFair(**params)

    elif model_name == "MinMaxFair_tpr":

        def model(**params):
            return models.MinMaxFair(fairness_constraint="tpr", **params)

    elif model_name == "MinMaxFairRegressor":

        def model(**params):
            return models.MinMaxFairRegressor(**params)

    elif model_name == "MinimaxPareto":

        def model(**params):
            return models.MinimaxPareto(**params)

    return model


def get_param_spaces(model_name):
    """Helper function to get parameter space from model name."""
    if model_name not in [
        "M2FGBClassifier_tpr",
        "M2FGBClassifier_v1",
        "M2FGBClassifier_pr",
        "FairGBMClassifier_eod",
        "MinMaxFair_tpr",
    ]:
        return models.PARAM_SPACES[model_name]
    elif "M2FGBClassifier" in model_name:
        return models.PARAM_SPACES["M2FGBClassifier"]
    elif model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES["FairGBMClassifier"]
    elif model_name == "MinMaxFair_tpr":
        return models.PARAM_SPACES["MinMaxFair"]


def get_param_spaces_acsincome(model_name):
    """Helper function to get parameter space from model name."""
    if model_name not in [
        "M2FGBClassifier_tpr",
        "M2FGBClassifier_pr",
        "FairGBMClassifier_eod",
        "MinMaxFair_tpr",
    ]:
        return models.PARAM_SPACES_ACSINCOME[model_name]
    elif model_name == "M2FGBClassifier_tpr" or model_name == "M2FGBClassifier_pr":
        return models.PARAM_SPACES_ACSINCOME["M2FGBClassifier"]
    elif model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES_ACSINCOME["FairGBMClassifier"]
    elif model_name == "MinMaxFair_tpr":
        return models.PARAM_SPACES_ACSINCOME["MinMaxFair"]


def get_param_list(param_space, n_params):

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


def eval_model(
    model_list,
    trials_df,
    thresh_type,
    X_train,
    Y_train,
    A_train,
    X_val,
    Y_val,
    A_val,
    X_test,
    Y_test,
    A_test,
):
    """Evaluate model performance and fairness metrics."""
    results_train = []
    results_val = []
    results_test = []

    # check if is classification
    if np.unique(Y_train).shape[0] == 2:
        is_classification = True
    else:
        is_classification = False

    def get_classif_metrics(y_true, y_pred, A):
        return {
            # perf metrics
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "acc": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            # fair metrics
            "eod": utils.equal_opportunity_score(y_true, y_pred, A),
            "spd": utils.statistical_parity_score(y_true, y_pred, A),
            "min_tpr": 1 - utils.min_true_positive_rate(y_true, y_pred, A),
            "min_pr": 1 - utils.min_positive_rate(y_true, y_pred, A),
            "min_bal_acc": 1 - utils.min_balanced_accuracy(y_true, y_pred, A),
            "min_acc": 1 - utils.min_accuracy(y_true, y_pred, A),
        }

    def get_reg_metrics(y_true, y_pred, A):
        return {
            "mse": np.mean((y_true - y_pred) ** 2),
            "max_mse": utils.max_mse(y_true, y_pred, A),
        }

    for m, model in tqdm(enumerate(model_list), total=len(model_list)):
        duration = trials_df.duration[trials_df.number == m].values[0]

        if is_classification:
            # get threshold
            if thresh_type == "ks":
                y_train_score = model.predict_proba(X_train)[:, 1]
                thresh = utils.get_best_threshold(Y_train, y_train_score)
            else:
                thresh = 0.5

            y_train_pred = model.predict_proba(X_train)[:, 1] > thresh
            y_val_pred = model.predict_proba(X_val)[:, 1] > thresh
            y_test_pred = model.predict_proba(X_test)[:, 1] > thresh

            results_train.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_classif_metrics(Y_train, y_train_pred, A_train),
                    "duration": duration,
                }
            )
            results_val.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_classif_metrics(Y_val, y_val_pred, A_val),
                    "duration": duration,
                }
            )
            results_test.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_classif_metrics(Y_test, y_test_pred, A_test),
                    "duration": duration,
                }
            )

        else:
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            thresh = 0.5
            results_train.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_reg_metrics(Y_train, y_train_pred, A_train),
                    "duration": duration,
                }
            )
            results_val.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_reg_metrics(Y_val, y_val_pred, A_val),
                    "duration": duration,
                }
            )
            results_test.append(
                {
                    "model": m,
                    "thresh": thresh,
                    **get_reg_metrics(Y_test, y_test_pred, A_test),
                    "duration": duration,
                }
            )

    results_train = pd.DataFrame(results_train)
    results_val = pd.DataFrame(results_val)
    results_test = pd.DataFrame(results_test)
    return results_train, results_val, results_test


def run_trial(trial, X_train, Y_train, A_train, X_val, Y_val, A_val, model_class, param_space, model_list):
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
    #model.fit(X_train, Y_train, A_train, X_val, Y_val, A_val)
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

    if args["dataset"] not in ["taiwan", "adult", "acsincome"]:
        param_space = get_param_spaces(args["model_name"])
    else:
        param_space = get_param_spaces_acsincome(args["model_name"])
    param_list = get_param_list(param_space, args["n_params"])
    # Load data
    X_train, A_train, Y_train, X_val, A_val, Y_val, X_test, A_test, Y_test = (
        data.get_strat_split(args["dataset"], args["n_groups"], 20, SEED)
    )

    study = optuna.create_study(direction="maximize")
    for param in param_list:
        study.enqueue_trial(param)
    model_list = []
    objective = lambda trial: run_trial(
        trial,
        X_train,
        Y_train,
        A_train,
        X_val, 
        Y_val, 
        A_val,
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
    trials_df.to_csv(os.path.join(args["output_dir"], f"trials.csv"), index=False)

    results_train, results_val, results_test = eval_model(
        model_list,
        trials_df,
        args["thresh"],
        X_train,
        Y_train,
        A_train,
        X_val,
        Y_val,
        A_val,
        X_test,
        Y_test,
        A_test,
    )

    # save results of fold
    results_train.to_csv(os.path.join(args["output_dir"], f"train.csv"), index=False)
    results_val.to_csv(os.path.join(args["output_dir"], f"val.csv"), index=False)
    results_test.to_csv(os.path.join(args["output_dir"], f"test.csv"), index=False)

    # save model
    with open(os.path.join(args["output_dir"], f"model.pkl"), "wb") as f:
        pkl.dump(model_list, f)



def experiment1(args):
    """Equalized loss experiment."""
    thresh = "ks"
    n_jobs = 10

    datasets = ["german", "compas", "enem"]
    n_groups_list = [8]
    model_name_list = [
        #"M2FGBClassifier",
        #"M2FGBClassifier",
        "M2FGBClassifier_tpr",
        #"FairGBMClassifier",
        #"MinMaxFair",
        "FairGBMClassifier_eod",
        #"MinMaxFair_tpr",
        #"LGBMClassifier",
        #"MinimaxPareto",
    ]

    n_params = args.n_params
    for dataset in datasets:
        for n_groups in n_groups_list:
            if dataset == "enem_large":
                n_groups = 27

            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if (
                        dataset == "acsincome"
                        or dataset == "taiwan"
                        or dataset == "adult"
                    ):
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now() - datetime.timedelta(hours=3)
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = (
                    f"../results_aaai/experiment_new/{dataset}_{n_groups}g/{model_name}"
                )
                config = {
                    "dataset": dataset,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "n_jobs": n_jobs,
                    "thresh": thresh,
                }
                run_subgroup_experiment(config)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now() - datetime.timedelta(hours=3)
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment_many_groups(fair_metric):
    """Equalized loss experiment."""
    n_folds = 10
    thresh = "ks"
    n_jobs = 10

    dataset = "enem_large"
    n_groups = 27
    model_name_list = [
        "M2FGBClassifier_tpr",
        # "M2FGBClassifier",
        "FairGBMClassifier",
        # "MinMaxFair",
        "LGBMClassifier",
        # "MinimaxPareto",
    ]

    n_params = 5
    for model_name in model_name_list:
        if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
            if dataset == "acsincome" or dataset == "taiwan" or dataset == "adult":
                n_params = 25

        with open("log.txt", "a+") as f:
            now = datetime.datetime.now() - datetime.timedelta(hours=3)
            f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

        output_dir = f"../results_aaai/experiment_{n_groups}g_{fair_metric}/{dataset}/{model_name}"
        args = {
            "dataset": dataset,
            "output_dir": output_dir,
            "model_name": model_name,
            "n_folds": n_folds,
            "n_groups": n_groups,
            "n_params": n_params,
            "n_jobs": n_jobs,
            "thresh": thresh,
        }
        run_subgroup_experiment(args)

        with open("log.txt", "a+") as f:
            now = datetime.datetime.now() - datetime.timedelta(hours=3)
            f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment_regression(args):
    """Equalized loss experiment."""
    thresh = "ks"
    n_jobs = 10

    dataset = "enem_reg"
    n_groups = 8
    model_name_list = [
        "MinMaxFairRegressor",
        #"LGBMRegressor",
        #"M2FGBRegressor",
    ]

    n_params = args.n_params
    for model_name in model_name_list:

        with open("log.txt", "a+") as f:
            now = datetime.datetime.now() - datetime.timedelta(hours=3)
            f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

        output_dir = f"../results_aaai/experiment_new/{dataset}_{n_groups}/{model_name}"
        args = {
            "dataset": dataset,
            "output_dir": output_dir,
            "model_name": model_name,
            "n_groups": n_groups,
            "n_params": n_params,
            "n_jobs": n_jobs,
            "thresh": thresh,
        }
        run_subgroup_experiment(args)

        with open("log.txt", "a+") as f:
            now = datetime.datetime.now() - datetime.timedelta(hours=3)
            f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def main():
    import lightgbm as lgb
    import fairgbm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_params", type=int, default=1000)

    lgb.register_logger(utils.CustomLogger())
    fairgbm.register_logger(utils.CustomLogger())

    # experiment1(parser.parse_args().fair_metric)
    # experiment_many_groups(parser.parse_args().fair_metric)
    #experiment_regression(parser.parse_args())
    experiment1(parser.parse_args())


if __name__ == "__main__":
    main()
