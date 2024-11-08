import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import os
import data
import models
import utils
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score


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
    args,
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
    model.fit(X_train, Y_train, A_train)

    Y_val_score = model.predict_proba(X_val)[:, 1]
    if args["thresh"] == "ks":
        thresh = utils.get_best_threshold(Y_val, Y_val_score)
    else:
        thresh = 0.5
    Y_val_pred = Y_val_score > thresh
    return scorer(Y_val, Y_val_pred, A_val)


def get_model(model_name, random_state=None):
    """Helper function to get model class from model name."""
    if model_name == "M2FGB":

        def model(**params):
            return models.M2FGB(random_state=random_state, **params)

    elif model_name == "M2FGB_grad":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient", random_state=random_state, **params
            )

    elif model_name == "M2FGB_eod":

        def model(**params):
            return models.M2FGB(
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_grad_eod":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient",
                fairness_constraint="equal_opportunity",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_spd":

        def model(**params):
            return models.M2FGB(
                fairness_constraint="demographic_parity",
                random_state=random_state,
                **params,
            )

    elif model_name == "M2FGB_grad_spd":

        def model(**params):
            return models.M2FGB(
                dual_learning="gradient",
                fairness_constraint="demographic_parity",
                random_state=random_state,
                **params,
            )

    elif model_name == "LGBMClassifier":

        def model(**params):
            return models.LGBMClassifier(random_state=random_state, verbose=-1, **params)

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

    return model


def get_param_spaces(model_name):
    """Helper function to get parameter space from model name."""
    if model_name not in [
        "M2FGB_eod",
        "M2FGB_spd",
        "M2FGB_grad_eod",
        "M2FGB_grad_spd",
        "FairGBMClassifier_eod",
        "FairClassifier_spd",
    ]:
        return models.PARAM_SPACES[model_name]
    elif model_name == "M2FGB_eod" or model_name == "M2FGB_spd":
        return models.PARAM_SPACES["M2FGB"]
    elif model_name == "M2FGB_grad_eod" or model_name == "M2FGB_grad_spd":
        return models.PARAM_SPACES["M2FGB_grad"]
    elif model_name == "FairGBMClassifier_eod":
        return models.PARAM_SPACES["FairGBMClassifier"]
    elif model_name == "FairClassifier_spd":
        return models.PARAM_SPACES["FairClassifier"]


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

    elif n_groups == 8:

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
            A_train = (
                X_train.sex.astype(str) + "_" + X_train.age.apply(age_cat).astype(str)
            )
            A_val = X_val.sex.astype(str) + "_" + X_val.age.apply(age_cat).astype(str)
            A_test = (
                X_test.sex.astype(str) + "_" + X_test.age.apply(age_cat).astype(str)
            )
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

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    print(sensitive_map)
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


def run_subgroup_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))
    results_train = []
    results_val = []
    results_test = []
    

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
        alpha=args["alpha"], performance_metric="bal_acc", fairness_metric=args["fairness_metric"]
    )

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
            args,
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=-1)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        model.fit(X_train, Y_train, A_train)

        y_train_score = model.predict_proba(X_train)[:, 1]
        y_val_score = model.predict_proba(X_val)[:, 1]
        y_test_score = model.predict_proba(X_test)[:, 1]
        if args["thresh"] == "ks":
            thresh = utils.get_best_threshold(Y_val, y_val_score)
        else:
            thresh = 0.5
        
        y_train_pred = y_train_score > thresh
        y_val_pred = y_val_score > thresh
        y_test_pred = y_test_score > thresh

        best_params["threshold"] = thresh
        joblib.dump(model, os.path.join(args["output_dir"], f"model_{i}.pkl"))
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")

        metrics_train = eval_model(Y_train, y_train_score, y_train_pred, A_train)
        metrics_val = eval_model(Y_val, y_val_score, y_val_pred, A_val)
        metrics_test = eval_model(Y_test, y_test_score, y_test_pred, A_test)
        
        results_train.append(metrics_train)
        results_val.append(metrics_val)
        results_test.append(metrics_test)
        
        
    results_train = pd.DataFrame(results_train)
    results_val = pd.DataFrame(results_val)
    results_test = pd.DataFrame(results_test)
    results_train.to_csv(os.path.join(args["output_dir"], "results_train.csv"), index=False)
    results_val.to_csv(os.path.join(args["output_dir"], "results_val.csv"), index=False)
    results_test.to_csv(os.path.join(args["output_dir"], "results.csv"), index=False)


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
        fairness_goal=args["goal"],
        M=1000,
        performance_metric="bal_acc",
        fairness_metric="eod",
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, SEED
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
            args,
        )
        study.optimize(objective, n_trials=args["n_trials"], n_jobs=-1)
        best_params = study.best_params.copy()

        model = model_class(**study.best_params)
        model.fit(X_train, Y_train, A_train)
        y_val_score = model.predict_proba(X_val)[:, 1]
        if args["thresh"] == "ks":
            thresh = utils.get_best_threshold(Y_val, y_val_score)
        else:
            thresh = 0.5
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


def experiment1():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairClassifier",
        "M2FGB",
        "M2FGB_grad",
    ]
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alphas += [0.05, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    alphas_adult = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for dataset in datasets:
        for alpha in alphas:
            if dataset == "adult" and not alpha in alphas_adult:
                continue

            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "output_dir": f"../results/group_experiment/{dataset}/{model_name}_{alpha}",
                    "model_name": model_name,
                    "n_trials": 50,
                    "n_groups": 2,
                    "fairness_metric" : "eod"
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup_experiment(args)


def experiment2():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairGBMClassifier_eod",
        "M2FGB",
        "M2FGB_grad",
        "M2FGB_eod",
        "M2FGB_grad_eod",
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
                    "n_groups": 4,
                    "fairness_metric" : "eod",
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup_experiment(args)


def experiment3():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairGBMClassifier_eod",
        "M2FGB",
        "M2FGB_grad",
        "M2FGB_eod",
        "M2FGB_grad_eod",
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
                    "n_groups": 8,
                    "fairness_metric" : "eod",
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup_experiment(args)


def experiment4():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "FairGBMClassifier_eod",
        "FairClassifier",
        "M2FGB",
        "M2FGB_grad",
        "M2FGB_eod",
        "M2FGB_grad_eod",
    ]
    goals = [0.95]
    for dataset in datasets:
        for goal in goals:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "goal": goal,
                    "output_dir": f"../results/fairness_goal_experiment2/{dataset}/{model_name}_{goal}",
                    "model_name": model_name,
                    "n_trials": 100,
                    "n_groups": 2,
                    "fairness_metric" : "eod",
                }
                print(f"{dataset} {model_name} {goal}")
                run_fairness_goal_experiment(args)


def experiment5():
    datasets = ["german", "compas", "adult"]
    model_names = [
        "LGBMClassifier",
        "FairClassifier_spd",
        "M2FGB_spd",
        "M2FGB_grad_spd",
    ]
    alphas = [0.7]

    for dataset in datasets:
        for alpha in alphas:
            for model_name in model_names:
                args = {
                    "dataset": dataset,
                    "alpha": alpha,
                    "output_dir": f"../results/group_experiment/{dataset}/{model_name}_{alpha}",
                    "model_name": model_name,
                    "n_trials": 50,
                    "n_groups": 2,
                    "fairness_metric" : "eod",
                }
                print(f"{dataset} {model_name} {alpha}")
                run_subgroup_experiment(args)


def main():
    experiment1()  # (binary groups)

    experiment2()  # (4 groups)

    experiment3()  # (8 groups)

    experiment4()  # (fairness goal)

    experiment5()  # (SPD)


if __name__ == "__main__":
    main()
