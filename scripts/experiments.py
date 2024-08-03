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
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    log_loss,
    recall_score
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
            return models.LGBMClassifier(
                random_state=random_state, **params
            )

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
    results_val = []
    results_test = []
    for m, model in tqdm(enumerate(model_list), total=len(model_list)):
        # get threshold
        if thresh_type == "ks":
            y_train_score = model.predict_proba(X_train)[:, 1]
            thresh = utils.get_best_threshold(Y_train, y_train_score)
        else:
            thresh = 0.5
        duration = trials_df.duration[trials_df.number == m].values[0]

        y_val_pred = model.predict_proba(X_val)[:, 1] > thresh
        y_test_pred = model.predict_proba(X_test)[:, 1] > thresh
        
        results_val.append({
            "model": m,
            "thresh": thresh,
            # perf metrics
            "bal_acc" : balanced_accuracy_score(Y_val, y_val_pred),
            "precision" : precision_score(Y_val, y_val_pred),
            "acc" : accuracy_score(Y_val, y_val_pred),
            "recall" : recall_score(Y_val, y_val_pred),
            # fair metrics
            "eod" : utils.equal_opportunity_score(Y_val, y_val_pred, A_val),
            "spd" : utils.statistical_parity_score(Y_val, y_val_pred, A_val),
            "min_tpr" : 1 - utils.min_true_positive_rate(Y_val, y_val_pred, A_val),
            "min_pr" : 1 - utils.min_positive_rate(Y_val, y_val_pred, A_val),
            "min_bal_acc" : 1 - utils.min_balanced_accuracy(Y_val, y_val_pred, A_val),
            "min_acc" : 1 - utils.min_accuracy(Y_val, y_val_pred, A_val),
            # time
            "duration" : duration
        })

        results_test.append({
            "model": m,
            "thresh": thresh,
            # perf metrics
            "bal_acc" : balanced_accuracy_score(Y_test, y_test_pred),
            "precision" : precision_score(Y_test, y_test_pred),
            "acc" : accuracy_score(Y_test, y_test_pred),
            "recall" : recall_score(Y_test, y_test_pred),
            # fair metrics
            "eod" : utils.equal_opportunity_score(Y_test, y_test_pred, A_test),
            "spd" : utils.statistical_parity_score(Y_test, y_test_pred, A_test),
            "min_tpr" : 1 - utils.min_true_positive_rate(Y_test, y_test_pred, A_test),
            "min_pr" : 1 - utils.min_positive_rate(Y_test, y_test_pred, A_test),
            "min_bal_acc" : 1 - utils.min_balanced_accuracy(Y_test, y_test_pred, A_test),
            "min_acc" : 1 - utils.min_accuracy(Y_test, y_test_pred, A_test),
            # time
            "duration" : duration
        })


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

    if args["dataset"] not in ["taiwan", "adult", "acsincome"]:
        param_space = get_param_spaces(args["model_name"])
    else:
        param_space = get_param_spaces_acsincome(args["model_name"])
    param_list = get_param_list(param_space, args["n_params"])

    for i in tqdm(range(args["n_folds"])):
        # Load and prepare data
        X_train, A_train, Y_train, X_val, A_val, Y_val, X_test, A_test, Y_test = data.get_fold(
            args["dataset"], i, args["n_folds"], args["n_groups"], SEED
        )

        
        
        study = optuna.create_study(
            direction="maximize"#, sampler=RandomSampler(seed=SEED)
        )
        for param in param_list:
            study.enqueue_trial(param)
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
        results_val.to_csv(
            os.path.join(args["output_dir"], f"validation_fold_{i}.csv"), index=False
        )
        results_test.to_csv(
            os.path.join(args["output_dir"], f"test_fold_{i}.csv"), index=False
        )


def experiment1(fair_metric):
    """Equalized loss experiment."""
    n_folds = 10
    thresh = "ks"
    n_jobs = 10

    datasets = [
        "german",
        "compas", 
        "taiwan", 
        "adult", 
        "enem"]
    n_groups_list = [8]#2, 4, 8]
    model_name_list = [
        "M2FGB_grad",
        "FairGBMClassifier",
        #"MinMaxFair",
        "LGBMClassifier",
        #"MinimaxPareto",
    ]

    n_params = 100
    for dataset in datasets:
        for n_groups in n_groups_list:
            for model_name in model_name_list:
                if model_name == "MinMaxFair" or model_name == "MinimaxPareto":
                    if dataset == "acsincome" or dataset == "taiwan" or dataset == "adult":
                        n_params = 25

                with open("log.txt", "a+") as f:
                    now = datetime.datetime.now()
                    f.write(f"Started: {dataset}, {n_groups}, {model_name} at {now}\n")

                output_dir = (
                    f"../results_aaai/experiment_{n_groups}g_{fair_metric}/{dataset}/{model_name}"
                )
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
                    now = datetime.datetime.now()
                    f.write(f"Finished: {dataset}, {n_groups}, {model_name} at {now}\n")


def main():
    import lightgbm as lgb
    import fairgbm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fair_metric", type=str, default="min_acc")

    lgb.register_logger(utils.CustomLogger())
    fairgbm.register_logger(utils.CustomLogger())

    experiment1(parser.parse_args().fair_metric)
   


if __name__ == "__main__":
    main()
