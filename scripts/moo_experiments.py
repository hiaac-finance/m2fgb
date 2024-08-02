import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

import optuna
from optuna.samplers import RandomSampler, TPESampler

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
            return models.LGBMClassifier(random_state=random_state, **params)

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


def eval_model(
    model_list,
    trials_df,
    thresh,
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
    results_val = []
    results_test = []
    for m, model in tqdm(enumerate(model_list), total=len(model_list)):
        # get threshold
        if thresh == "ks":
            y_train_score = model.predict_proba(X_train)[:, 1]
            thresh = utils.get_best_threshold(Y_train, y_train_score)
        else:
            thresh = 0.5
        duration = trials_df.duration[trials_df.number == m].values[0]

        y_val_pred = model.predict_proba(X_val)[:, 1] > thresh
        y_test_pred = model.predict_proba(X_test)[:, 1] > thresh

        results_val.append(
            {
                "model": m,
                "thresh": thresh,
                # perf metrics
                "bal_acc": balanced_accuracy_score(Y_val, y_val_pred),
                "precision": precision_score(Y_val, y_val_pred),
                "acc": accuracy_score(Y_val, y_val_pred),
                "recall": recall_score(Y_val, y_val_pred),
                # fair metrics
                "eod": utils.equal_opportunity_score(Y_val, y_val_pred, A_val),
                "spd": utils.statistical_parity_score(Y_val, y_val_pred, A_val),
                "min_tpr": 1 - utils.min_true_positive_rate(Y_val, y_val_pred, A_val),
                "min_pr": 1 - utils.min_positive_rate(Y_val, y_val_pred, A_val),
                "min_bal_acc": 1
                - utils.min_balanced_accuracy(Y_val, y_val_pred, A_val),
                "min_acc": 1 - utils.min_accuracy(Y_val, y_val_pred, A_val),
                # time
                "duration": duration,
            }
        )

        results_test.append(
            {
                "model": m,
                "thresh": thresh,
                # perf metrics
                "bal_acc": balanced_accuracy_score(Y_test, y_test_pred),
                "precision": precision_score(Y_test, y_test_pred),
                "acc": accuracy_score(Y_test, y_test_pred),
                "recall": recall_score(Y_test, y_test_pred),
                # fair metrics
                "eod": utils.equal_opportunity_score(Y_test, y_test_pred, A_test),
                "spd": utils.statistical_parity_score(Y_test, y_test_pred, A_test),
                "min_tpr": 1
                - utils.min_true_positive_rate(Y_test, y_test_pred, A_test),
                "min_pr": 1 - utils.min_positive_rate(Y_test, y_test_pred, A_test),
                "min_bal_acc": 1
                - utils.min_balanced_accuracy(Y_test, y_test_pred, A_test),
                "min_acc": 1 - utils.min_accuracy(Y_test, y_test_pred, A_test),
                # time
                "duration": duration,
            }
        )

    results_val = pd.DataFrame(results_val)
    results_test = pd.DataFrame(results_test)
    return results_val, results_test


def run_trial(
    trial,
    model_class,
    param_space,
    perf_metric,
    fair_metric,
    fold_data
):
    """Function to run a single trial of optuna."""
    n_folds = len(fold_data)
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
    perf_list = []
    perf_test_list = []
    fair_list = []
    fair_test_list = []
    for fold in range(n_folds):
        model = model_class(**params)
        (
            X_train,
            A_train,
            Y_train,
            X_val,
            A_val,
            Y_val,
            X_test,
            A_test,
            Y_test,
        ) = fold_data[fold]
        model.fit(X_train, Y_train, A_train)
        thresh = utils.get_best_threshold(Y_train, model.predict_proba(X_train)[:, 1])
        Y_train_pred = model.predict_proba(X_train)[:, 1] > thresh
        Y_val_pred = model.predict_proba(X_val)[:, 1] > thresh
        Y_test_pred = model.predict_proba(X_test)[:, 1] > thresh
        perf_list.append(perf_metric(Y_val, Y_val_pred))
        fair_list.append(fair_metric(Y_val, Y_val_pred, A_val))
        perf_test_list.append(perf_metric(Y_test, Y_test_pred))
        fair_test_list.append(fair_metric(Y_test, Y_test_pred, A_test))

    # save val and test accuracy in the trial
    trial.set_user_attr("perf_val", np.mean(perf_list))
    trial.set_user_attr("perf_test", np.mean(perf_test_list))
    trial.set_user_attr("fair_val", np.mean(fair_list))
    trial.set_user_attr("fair_test", np.mean(fair_test_list))

    return np.mean(perf_list), np.mean(fair_list)


def run_subgroup_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))

    fold_data = []
    for fold in range(args["n_folds"]):
        fold_data.append(
            data.get_fold_holdout(
                args["dataset"], fold, args["n_folds"], args["n_groups"], SEED
            )
        )

    # X_train, A_train, Y_train, X_val, A_val, Y_val, X_test, A_test, Y_test = data.get_fold(
    #    args["dataset"], i, args["n_folds"], args["n_groups"], SEED
    # )

    if args["dataset"] not in ["taiwan", "adult", "acsincome"]:
        param_space = get_param_spaces(args["model_name"])
    else:
        param_space = get_param_spaces_acsincome(args["model_name"])

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=TPESampler(seed=SEED, n_startup_trials=50),
    )
    if args["fair_metric"] == "min_acc":
        fair_metric = utils.min_accuracy
    elif args["fair_metric"] == "min_bal_acc":
        fair_metric = utils.min_balanced_accuracy
    elif args["fair_metric"] == "min_pr":
        fair_metric = utils.min_positive_rate
    elif args["fair_metric"] == "min_tpr":
        fair_metric = utils.min_true_positive_rate

    objective = lambda trial: run_trial(
        trial,
        get_model(args["model_name"]),
        param_space,
        balanced_accuracy_score,
        fair_metric,
        args["dataset"],
        args["n_folds"],
        args["n_groups"],
    )
    study.optimize(
        objective,
        n_trials=args["n_params"],
        n_jobs=args["n_jobs"],
        show_progress_bar=True,
    )
    trials_df = study.trials_dataframe(
        attrs=("number", "duration", "params", "user_attrs")
    )
    trials_df.to_csv(os.path.join(args["output_dir"], f"trials.csv"), index=False)


def experiment1():
    """Equalized loss experiment."""
    n_folds = 5
    thresh = "ks"
    n_jobs = 10

    datasets = ["german", "compas", "taiwan", "adult"]  # , "acsincome"]
    n_groups_list = [8]  # 2, 4, 8]
    model_name_list = [
        "M2FGB_grad",
        # "FairGBMClassifier",
        # "MinMaxFair",
        "LGBMClassifier",
        # "MinimaxPareto",
    ]
    fair_metric = "min_acc"
    n_params = 250
    for dataset in datasets:
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if (
                        dataset == "acsincome"
                        or dataset == "taiwan"
                        or dataset == "adult"
                    ):
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = f"../results_aaai_v2/experiment_{n_groups}_groups/{dataset}/{model_name}"
                args = {
                    "dataset": dataset,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_folds": n_folds,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "n_jobs": n_jobs,
                    "fair_metric": fair_metric,
                    "thresh": thresh,
                }
                run_subgroup_experiment(args)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment2():
    """Positive rate experiment."""
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
    n_groups_list = [4, 8]
    model_name_list = [
        "M2FGB_grad_pr",
        "LGBMClassifier",
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
    """Equalized loss experiment."""
    n_folds = 5
    thresh = "ks"
    n_jobs = 10

    datasets = ["german", "compas", "taiwan", "adult"]  # , "acsincome"]
    n_groups_list = [8]  # 2, 4, 8]
    model_name_list = [
        "M2FGB_grad_tpr",
        # "FairGBMClassifier",
        # "MinMaxFair",
        "LGBMClassifier",
        # "MinimaxPareto",
    ]
    fair_metric = "min_tpr"

    n_params = 250
    for dataset in datasets:
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if (
                        dataset == "acsincome"
                        or dataset == "taiwan"
                        or dataset == "adult"
                    ):
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = f"../results_aaai_v2/experiment_{n_groups}_groups/{dataset}/{model_name}"
                args = {
                    "dataset": dataset,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "n_folds": n_folds,
                    "n_groups": n_groups,
                    "n_params": n_params,
                    "n_jobs": n_jobs,
                    "thresh": thresh,
                    "fair_metric": fair_metric,
                }
                run_subgroup_experiment(args)

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def experiment4():
    """Experiment that consider multiple fair_weights values and fit random model with each."""
    if not os.path.exists("../results/experiment_fair_weight"):
        os.mkdir("../results/experiment_fair_weight")

    n_folds = 10
    fold = 0
    n_groups = 4
    n_params = 100
    fair_weight_list = [
        0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1,
    ]
    datasets = ["german", "compas", "acsincome"]

    param_space = models.PARAM_SPACES["M2FGB_grad"].copy()
    del param_space["fair_weight"]
    param_list = get_param_list(param_space, n_params)

    for dataset in datasets:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            dataset, fold, n_folds, SEED
        )
        A_train, A_val, A_test = get_subgroup_feature(
            dataset, X_train, X_val, X_test, n_groups
        )
        X_train, X_val, X_test = data.preprocess_dataset(
            dataset, X_train, X_val, X_test
        )

        results = []
        for fair_weight in fair_weight_list:

            for p_i in range(n_params):
                param_list[p_i]["fair_weight"] = fair_weight

            model_list = []
            study = optuna.create_study(
                direction="maximize", sampler=RandomSampler(seed=SEED)
            )
            objective = lambda trial: run_trial_fixed(
                trial,
                X_train,
                Y_train,
                A_train,
                get_model("M2FGB_grad", random_state=SEED),
                param_list,
                model_list,
            )
            study.optimize(
                objective,
                n_trials=n_params,
                n_jobs=10,
                show_progress_bar=True,
            )

            for i, model in enumerate(model_list):

                Y_pred = model.predict_proba(X_train)[:, 1]
                overall_score = log_loss(Y_train, Y_pred)
                group_scores = utils.logloss_group(
                    Y_train, Y_pred, A_train, "equalized_loss"
                )

                results.append(
                    {
                        "param": i,
                        "fair_weight": fair_weight,
                        "overall_score": overall_score,
                        "max_group_score": group_scores.max(),
                    }
                    | {
                        f"group_{i}": group_score
                        for i, group_score in enumerate(group_scores)
                    }
                    | {
                        f"param_{key}": value
                        for key, value in model.get_params().items()
                    }
                )

            pd.DataFrame(results).to_csv(
                f"../results/experiment_fair_weight/{dataset}.csv", index=False
            )


def main():
    import lightgbm as lgb
    import fairgbm

    lgb.register_logger(utils.CustomLogger())
    fairgbm.register_logger(utils.CustomLogger())

    experiment1()
    # experiment2()
    experiment3()
    # experiment4()


if __name__ == "__main__":
    main()
