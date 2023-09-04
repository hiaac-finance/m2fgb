import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score


def get_best_threshold(y_ground, y_pred):
    fpr, tpr, thresholds = roc_curve(y_ground, y_pred)
    return thresholds[np.argmax(tpr - fpr)]


def fnr_score(y_ground, y_pred):
    fn = ((y_pred == 0) & (y_ground == 1)).sum()
    tp = ((y_pred == 1) & (y_ground == 1)).sum()
    fnr = fn / (fn + tp)
    return fnr


def tpr_score(y_ground, y_pred):
    fn = ((y_pred == 0) & (y_ground == 1)).sum()
    tp = ((y_pred == 1) & (y_ground == 1)).sum()
    tpr = tp / (tp + fn)
    return tpr

def logloss_score(y_ground, y_pred):
    return -np.mean(y_ground * np.log(y_pred) + (1 - y_ground) * np.log(1 - y_pred))


def eval_model_data(y_ground, y_pred, p=None, name=""):
    roc = roc_auc_score(y_ground, y_pred)
    if p is None:
        p = get_best_threshold(y_ground, y_pred)
    logloss = logloss_score(y_ground, y_pred)
    fnr = fnr_score(y_ground, y_pred > p)
    tpr = tpr_score(y_ground, y_pred > p)
    acc = accuracy_score(y_ground, y_pred > p)
    precision = precision_score(y_ground, y_pred > p)
    return [
        {
            "roc": roc,
            "tpr": tpr,
            "fnr": fnr,
            "logloss": logloss,
            "threshold": p,
            "accuracy": acc,
            "precision": precision,
            "name": name,
        }
    ]


def eval_model_subgroups(y_ground, y_pred, subgroup, p=None, name=""):
    g_ = np.unique(subgroup)
    results = []
    for g in g_:
        results += eval_model_data(
            y_ground[subgroup == g], y_pred[subgroup == g], p, f"{name}_g{g}"
        )
    return results


def eval_model_train_test(
    model, X_train, Y_train, subgroup_train, X_test, Y_test, subgroup_test
):
    results = []
    Y_train_pred = model.predict_proba(X_train)[:, 1]
    p = get_best_threshold(Y_train, Y_train_pred)
    results += eval_model_subgroups(Y_train, Y_train_pred, subgroup_train, p, "train")
    Y_test_pred = model.predict_proba(X_test)[:, 1]
    results += eval_model_subgroups(Y_test, Y_test_pred, subgroup_test, p, "test")
    return pd.DataFrame(results)


def plot_metric_lambda(results, metric, axs=None):
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), sharey=True)

    for i, ds in enumerate(["train", "test"]):
        results_ = results[results.name.str.contains(ds)]
        results_0 = results_[results_.name.str.contains("0")]
        results_1 = results_[results_.name.str.contains("1")]
        axs[i].plot(results_0["lambda"], results_0[metric], label=f"{ds}_0")
        axs[i].plot(results_1["lambda"], results_1[metric], label=f"{ds}_1")
        axs[i].set_xlabel("lambda")
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].set_xscale("symlog", linthresh=0.01)
        axs[i].set_title("Performance in " + ds)
        axs[i].grid(True)
    return


def comparison_subgrous_metrics_lambda(results):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharey="row")
    plot_metric_lambda(results, "logloss", axs[0])
    plot_metric_lambda(results, "roc", axs[1])
    plot_metric_lambda(results, "tpr", axs[2])
    

    plt.tight_layout()
