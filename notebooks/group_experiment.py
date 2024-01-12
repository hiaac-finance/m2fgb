from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import pandas as pd
from tqdm import tqdm
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)
import sys
import os
import glob

from fairgbm import FairGBMClassifier
from lightgbm import LGBMClassifier


sys.path.append("../scripts")
import data
import models
import utils

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


def get_group_feature(dataset, X_train, X_val, X_test):
    if dataset == "german":
        A_train = X_train.Gender.astype(str)
        A_val = X_val.Gender.astype(str)
        A_test = X_test.Gender.astype(str)
    elif dataset == "adult":
        A_train = X_train.sex.astype(str)
        A_val = X_val.sex.astype(str)
        A_test = X_test.sex.astype(str)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def eval_model(y_true, y_score, y_pred, A):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_score)
    eq_loss = utils.equalized_loss_score(y_true, y_score, A)
    eod = utils.equal_opportunity_score(y_true, y_pred, A)
    spd = utils.statistical_parity_score(y_true, y_pred, A)
    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "roc": roc,
        "eq_loss": eq_loss,
        "eod": eod,
        "spd": spd,
    }


def get_model(model_name, random_state=None):
    if model_name == "XtremeFair":

        def model(**params):
            return models.XtremeFair(random_state=random_state, **params)

    elif model_name == "XtremeFair_grad":

        def model(**params):
            return models.XtremeFair(
                dual_learning="gradient", random_state=random_state, **params
            )

    elif model_name == "LGBMClassifier":

        def model(**params):
            return LGBMClassifier(random_state=random_state, verbose=-1, **params)

    elif model_name == "FairGBMClassifier":

        def model(**params):
            return FairGBMClassifier(random_state=random_state, **params)

    elif model_name == "ExponentiatedGradient":

        def model(**params):
            return models.ExponentiatedGradient_Wrap(
                random_state=random_state, **params
            )

    return model


def get_param_spaces(model_name):
    if model_name == "XtremeFair":
        return models.PARAM_SPACES["XtremeFair"]
    elif model_name == "XtremeFair_grad":
        return models.PARAM_SPACES["XtremeFair_grad"]
    elif model_name == "LGBMClassifier":
        return models.PARAM_SPACES["LGBMClassifier"]
    elif model_name == "FairGBMClassifier":
        return models.PARAM_SPACES["FairGBMClassifier"]
    elif model_name == "ExponentiatedGradient":
        return models.PARAM_SPACES["ExponentiatedGradient"]


def group_experiment(args):
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
    results.to_csv(os.path.join(args["output_dir"], "results.csv"))


def summarize(dataset_name):
    experiments = glob.glob(f"../results/group_experiment/{dataset_name}/*")
    results = []
    for experiment in experiments:
        df = pd.read_csv(os.path.join(experiment, "results.csv"))
        df["experiment"] = "_".join(experiment.split("/")[-1].split("_")[:-1])
        df["alpha"] = float(experiment.split("/")[-1].split("_")[-1])
        df["eq_loss"] = df["eq_loss"].abs()
        df["spd"] = 1 - df["spd"].abs()
        df["eod"] = 1 - df["eod"].abs()
        results.append(df.iloc[:, 1:])
    results = pd.concat(results)
    # for each experiment, calculate the mean and std of each metric
    results_mean = results.groupby(["experiment", "alpha"]).mean()
    # results_std = results.groupby(["experiment", "alpha"]).std()

    # combine dataframes into one with reorganized columns
    # results = pd.concat([results_mean, results_std], axis=1)
    # results.columns = pd.MultiIndex.from_product(
    #    [["mean", "std"], results_mean.columns]
    # )
    # results = results.swaplevel(axis=1)
    results = results_mean
    results = results[["bal_acc", "eod"]]
    results = results.round(3)
    results = results.reset_index()

    # verify which points are not dominated
    dominated = []
    for i, row in results.iterrows():
        is_dominated = False
        for j, row2 in results.iterrows():
            if i == j:
                continue
            if row["bal_acc"] <= row2["bal_acc"] and row["eod"] <= row2["eod"]:
                is_dominated = True
                break
        dominated.append(is_dominated)
    results["dominated"] = dominated
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i, model_name in enumerate(results["experiment"].unique()):
        df = results[results["experiment"] == model_name]
        ax.scatter(
            df["bal_acc"],
            df["eod"],
            s=[20 if dominated else 50 for dominated in df["dominated"]],
            label=model_name,
        )

    ax.legend()
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("1 - Equal Opportunity Difference")
    ax.set_title(dataset_name)
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"../results/group_experiment/{dataset_name}_plot.jpg")


def main():
    datasets = ["adult"]
    model_names = [
        "LGBMClassifier",
        "FairGBMClassifier",
        "XtremeFair",
        "XtremeFair_grad",
    ]
    alphas = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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
                group_experiment(args)

        print(summarize(dataset))


if __name__ == "__main__":
    main()
