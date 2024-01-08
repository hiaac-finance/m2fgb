import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score


def get_best_threshold(y_ground, y_pred):
    """Calculates the threshold according to the KS statistic"""
    fpr, tpr, thresholds = roc_curve(y_ground, y_pred)
    return thresholds[np.argmax(tpr - fpr)]


def fnr_score(y_ground, y_pred):
    """Calculates the false negative rate"""
    fn = ((y_pred == 0) & (y_ground == 1)).sum()
    tp = ((y_pred == 1) & (y_ground == 1)).sum()
    fnr = fn / (fn + tp)
    return fnr


def tpr_score(y_ground, y_pred):
    """Calculates the true positive rate"""
    fn = ((y_pred == 0) & (y_ground == 1)).sum()
    tp = ((y_pred == 1) & (y_ground == 1)).sum()
    tpr = tp / (tp + fn)
    return tpr


def equal_opportunity_score(y_ground, y_pred, A):
    """Calculate the difference between true poisitive rates of the groups.
    If A has two values, it must be 0 and 1, and it can also be applied to more than two groups (the result is the difference between the max value and min value).

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted probabilities of the positive class
    A : ndarray
        Group labels

    Returns
    -------
    float
        Equal opportunity score score
    """
    if len(np.unique(A)) > 2:
        max_ = -np.inf
        min_ = np.inf
        for a in np.unique(A):
            max_ = max(max_, np.mean(y_pred[(A == a) & (y_ground == 1)]))
            min_ = min(min_, np.mean(y_pred[(A == a) & (y_ground == 1)]))
        return max_ - min_

    return np.mean(y_pred[(A == 1) & (y_ground == 1)]) - np.mean(
        y_pred[(A == 0) & (y_ground == 1)]
    )


def statistical_parity_score(y_ground, y_pred, A):
    """Calculate the difference between probability of true outcome of the groups.
    If A has two values, it must be 0 and 1, and it can also be applied to more than two groups (the result is the difference between the max value and min value).

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted probabilities of the positive class
    A : ndarray
        Group labels

    Returns
    -------
    float
        Statistical parity score
    """
    if len(np.unique(A)) > 2:
        max_ = -np.inf
        min_ = np.inf
        for a in np.unique(A):
            max_ = max(max_, np.mean(y_pred[A == a]))
            min_ = min(min_, np.mean(y_pred[A == a]))
        return max_ - min_

    return np.mean(y_pred[A == 1]) - np.mean(y_pred[A == 0])


def get_combined_metrics_scorer(
    alpha=1, performance_metric="acc", fairness_metric="eod"
):
    def scorer(y_ground, y_pred, A):
        if performance_metric == "acc":
            perf = accuracy_score(y_ground, y_pred)
        elif performance_metric == "roc_auc":
            perf = roc_auc_score(y_ground, y_pred)

        if fairness_metric == "eod":
            fair = equal_opportunity_score(y_ground, y_pred, A)
        elif fairness_metric == "spd":
            fair = statistical_parity_score(y_ground, y_pred, A)

        return alpha * perf + (1 - alpha) * abs(fair)

    return scorer


def equalized_loss_score(y_ground, y_prob, A):
    """Calculate the difference between the mean loss of groups. The loss is binary cross entropy.
    If A has two values, it must be 0 and 1, and it can also be applied to more than two groups (the result is the difference between the max loss and min loss).

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted probabilities of the positive class
    A : ndarray
        Group labels

    Returns
    -------
    float
        Difference between the mean loss of groups
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -(y_ground * np.log(y_prob) + (1 - y_ground) * np.log(1 - y_prob))
    if len(np.unique(A)) > 2:
        max_ = -np.inf
        min_ = np.inf
        for a in np.unique(A):
            max_ = max(max_, np.mean(loss[A == a]))
            min_ = min(min_, np.mean(loss[A == a]))
        return max_ - min_

    return np.mean(loss[A == 1]) - np.mean(loss[A == 0])


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
    model, X_train, Y_train, subgroup_train, X_test, Y_test, subgroup_test, p=None
):
    results = []
    Y_train_pred = model.predict_proba(X_train)[:, 1]
    if p is None:
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
        axs[i].set_xlabel("Fairness weight")
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].set_xscale("symlog", linthresh=0.01)
        axs[i].set_title("Performance in " + ds)
        axs[i].grid(True)
    return


def plot_metric_diff_lambda(results, metric, axs=None):
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))

    color_map = {"train": "b", "test": "r"}
    for i, ds in enumerate(["train", "test"]):
        results_ = results[results.name.str.contains(ds)]
        results_0 = results_[results_.name.str.contains("0")]
        results_1 = results_[results_.name.str.contains("1")]
        axs.set_xscale("symlog", linthresh=0.01)
        axs.plot(
            results_0["lambda"],
            results_0[metric].values - results_1[metric].values,
            label=f"{ds}",
            color=color_map[ds],
            lw=2,
        )
        axs.set_xlabel("Fairness weight")
        axs.set_ylabel(metric + " difference")
        axs.legend()
        axs.grid(True)

    # plot the zero line
    axs.plot(
        results_0["lambda"],
        np.zeros(results_0["lambda"].shape),
        color="k",
        ls="--",
        lw=1,
    )
    return


def comparison_subgrous_metrics_lambda(results):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharey="row")
    plot_metric_lambda(results, "logloss", axs[0])
    plot_metric_lambda(results, "accuracy", axs[1])
    plot_metric_lambda(results, "tpr", axs[2])

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharey="row")
    plot_metric_diff_lambda(results, "logloss", axs[0])
    plot_metric_diff_lambda(results, "accuracy", axs[1])
    plot_metric_diff_lambda(results, "tpr", axs[2])
    plt.show()
