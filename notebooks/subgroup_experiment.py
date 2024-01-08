from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from tqdm import tqdm
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
import sys
import os
import glob

from fairgbm import FairGBMClassifier


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
    param_space
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
    else:
        model.fit(X_train, Y_train, A_train)
    Y_val_pred = model.predict(X_val)
    return scorer(Y_val, Y_val_pred, A_val)


def get_subgroup_feature(dataset, X_train, X_val, X_test):
    if dataset == "german2":
        A_train = X_train.Gender.astype(str) + "_" + (X_train.Age > 50).astype(str)
        A_val = X_val.Gender.astype(str) + "_" + (X_val.Age > 50).astype(str)
        A_test = X_test.Gender.astype(str) + "_" + (X_test.Age > 50).astype(str)
    elif dataset == "adult":
        A_train = X_train.sex.astype(str) + "_" + (X_train.age > 50).astype(str)
        A_val = X_val.sex.astype(str) + "_" + (X_val.age > 50).astype(str)
        A_test = X_test.sex.astype(str) + "_" + (X_test.age > 50).astype(str)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A_train.unique())])
    A_train = A_train.map(sensitive_map)
    A_val = A_val.map(sensitive_map)
    A_test = A_test.map(sensitive_map)
    return A_train, A_val, A_test


def eval_model(y_ground, y_prob, y_pred, A):
    acc = accuracy_score(y_ground, y_pred)
    roc = roc_auc_score(y_ground, y_prob)
    eq_loss = utils.equalized_loss_score(y_ground, y_prob, A)
    eod = utils.equal_opportunity_score(y_ground, y_pred, A)
    spd = utils.statistical_parity_score(y_ground, y_pred, A)
    return {"acc": acc, "roc": roc, "eq_loss": eq_loss, "eod": eod, "spd": spd}


def get_model(model_name, random_state=None):
    if model_name == "XtremeFair":

        def model(**params):
            return models.XtremeFair(random_state=random_state, **params)

    elif model_name == "XtremeFair_grad":

        def model(**params):
            return models.XtremeFair(
                dual_learning="gradient", random_state=random_state, **params
            )

    elif model_name == "XGBClassifier":

        def model(**params):
            assert params["fair_weight"] == 0
            return models.XtremeFair(random_state=random_state, **params)

    elif model_name == "FairGBMClassifier":

        def model(**params):
            return FairGBMClassifier(random_state=random_state, **params)

    return model


def get_param_spaces(model_name):
    if model_name == "XtremeFair":
        return models.PARAM_SPACES["XtremeFair"]
    elif model_name == "XtremeFair_grad":
        return models.PARAM_SPACES["XtremeFair"]
    elif model_name == "XGBClassifier":
        return models.PARAM_SPACES["XGBClassifier"]
    elif model_name == "FairGBMClassifier":
        return models.PARAM_SPACES["FairGBMClassifier"]


def subgroup_experiment(args):
    # create output directory if not exists
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    # clear best_params.txt if exists
    if os.path.exists(os.path.join(args["output_dir"], f"best_params.txt")):
        os.remove(os.path.join(args["output_dir"], f"best_params.txt"))

    results = []

    cat_features = data.CAT_FEATURES[args["dataset"]]
    num_features = data.NUM_FEATURES[args["dataset"]]
    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), num_features),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                cat_features,
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")
    scorer = utils.get_combined_metrics_scorer(
        alpha=args["alpha"], performance_metric="acc", fairness_metric="eod"
    )

    for i in tqdm(range(10)):
        # Load and prepare data
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_fold(
            args["dataset"], i, 0
        )

        # Define sensitive attribute from gender and age
        A_train, A_val, A_test = get_subgroup_feature(
            args["dataset"], X_train, X_val, X_test
        )

        preprocess = Pipeline([("preprocess", col_trans)])
        preprocess.fit(X_train)
        X_train = preprocess.transform(X_train)
        X_val = preprocess.transform(X_val)
        X_test = preprocess.transform(X_test)

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
            get_model(args["model_name"], random_state = SEED),
            get_param_spaces(args["model_name"]),
        )
        study.optimize(objective, n_trials=args["n_trials"])
        best_params = study.best_params.copy()


        model = get_model(args["model_name"])(**study.best_params)
        if isinstance(model, FairGBMClassifier):
            model.fit(X_train, Y_train, constraint_group=A_train)
        else:
            model.fit(X_train, Y_train, A_train)
        y_prob = model.predict_proba(X_train)[:, 1]
        thresh = utils.get_best_threshold(Y_train, y_prob)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = y_prob_test > thresh
        best_params["threshold"] = thresh

        # save best params
        with open(os.path.join(args["output_dir"], f"best_params.txt"), "a+") as f:
            f.write(str(best_params))
            f.write("\n")

        metrics = eval_model(Y_test, y_prob_test, y_pred_test, A_test)
        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args["output_dir"], "results.csv"))
    results.mean().to_csv(os.path.join(args["output_dir"], "results_mean.csv"))


def summarize(dataset_name):
    experiments = glob.glob(f"../results/subgroup_experiment/{dataset_name}/*")
    results = []
    for experiment in experiments:
        df = pd.read_csv(os.path.join(experiment, "results.csv"))
        df["experiment"] = experiment.split("/")[-1]
        df["eq_loss"] = 1 - df["eq_loss"].abs()
        df["spd"] = 1 - df["spd"].abs()
        df["eod"] = 1 - df["eod"].abs()
        results.append(df.iloc[:, 1:])
    results = pd.concat(results)
    # for each experiment, calculate the mean and std of each metric
    results_mean = results.groupby("experiment").mean()
    results_std = results.groupby("experiment").std()

    # combine dataframes into one with reorganized columns
    results = pd.concat([results_mean, results_std], axis=1)
    results.columns = pd.MultiIndex.from_product(
        [["mean", "std"], results_mean.columns]
    )
    results = results.swaplevel(axis=1)
    results = results[["roc", "acc", "eod", "eq_loss", "spd"]]
    results = results.round(3)
    print(results)


for dataset in ["german2", "adult"]:
    continue
    for alpha in [1, 0.75]:
        for model_name in [
            "XtremeFair",
            "XtremeFair_grad",
            "XGBClassifier",
            "FairGBMClassifier",
        ]:
            args = {
                "dataset": dataset,
                "alpha": alpha,
                "output_dir": f"../results/subgroup_experiment/{dataset}/{model_name}_{alpha}",
                "model_name": model_name,
                "n_trials": 100,
            }
            subgroup_experiment(args)

        print(summarize(dataset))
