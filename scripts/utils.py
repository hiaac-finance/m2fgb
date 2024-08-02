import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
)

import logging


class CustomLogger(logging.Logger):
    """Custom logger to suppress warnings from LightGBM (due to a bug, verbose does not supress)"""

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

def fpr_score(y_ground, y_pred):
    """Calculates the false positive rate"""
    fp = ((y_pred == 1) & (y_ground == 0)).sum()
    tn = ((y_pred == 0) & (y_ground == 0)).sum()
    fpr = fp / (fp + tn)
    return fpr

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


def min_true_positive_rate(y_ground, y_pred, A):
    """Calculate the minimum true positive rate of the groups.
    Return 1 - tpr so that the lowest the better.
    It work with multiple groups.

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
    min_tpr = np.inf
    for a in np.unique(A):
        tpr = np.mean(y_pred[(A == a) & (y_ground == 1)])
        min_tpr = min(min_tpr, tpr)
    return 1 - min_tpr

def max_fnr_score(y_ground, y_pred, A):
    max_fnr = -np.inf
    for a in np.unique(A):
        # check if the group has any positive class
        if np.sum(y_ground[A == a]) == 0:
            continue
        fnr = fnr_score(y_ground[A == a], y_pred[A == a])
        max_fnr = max(max_fnr, fnr)
    return max_fnr

def max_fpr_score(y_ground, y_pred, A):
    max_fpr = -np.inf
    for a in np.unique(A):
        # check if the group has any negative class
        if np.sum(1 - y_ground[A == a]) == 0:
            continue
        fpr = fpr_score(y_ground[A == a], y_pred[A == a])
        max_fpr = max(max_fpr, fpr)
    return max_fpr

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


def min_positive_rate(y_ground, y_pred, A):
    """Calculate the minimum probability of true outcome of the groups.
    It work with multiple groups.

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
    min_pr = np.inf

    for a in np.unique(A):
        pr = np.mean(y_pred[A == a])
        min_pr = min(min_pr, pr)
    return 1 - min_pr


def min_balanced_accuracy(y_ground, y_pred, A):
    """Calculate the minimum balanced accuracy among groups.
    It will return 1 - bal_acc so that values close to 0 are better.
    It work with multiple groups.

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted class
    A : ndarray
        Group labels

    Returns
    -------
    float
        1 - Minimum balanced accuracy
    """
    min_bal_acc = np.inf
    for a in np.unique(A):
        bal_acc = balanced_accuracy_score(y_ground[A == a], y_pred[A == a])
        min_bal_acc = min(min_bal_acc, bal_acc)
    return 1 - min_bal_acc


def min_accuracy(y_ground, y_pred, A):
    """Calculate the minimum accuracy among groups.
    It will return 1 - acc so that values close to 0 are better.
    It work with multiple groups.

    Parameters
    ----------
    y_ground : ndarray
        Ground truth labels in {0, 1}
    y_prob : ndarray
        Predicted class
    A : ndarray
        Group labels

    Returns
    -------
    float
        1 - Minimum accuracy
    """
    min_acc = np.inf
    for a in np.unique(A):
        acc = accuracy_score(y_ground[A == a], y_pred[A == a])
        min_acc = min(min_acc, acc)
    return 1 - min_acc


def get_combined_metrics_scorer(
    alpha=1, performance_metric="acc", fairness_metric="eod"
):
    def scorer(y_ground, y_pred, A):
        if performance_metric == "acc":
            perf = accuracy_score(y_ground, y_pred)
        elif performance_metric == "bal_acc":
            perf = balanced_accuracy_score(y_ground, y_pred)
        elif performance_metric == "roc_auc":
            perf = roc_auc_score(y_ground, y_pred)

        if fairness_metric == "eod":
            fair = equal_opportunity_score(y_ground, y_pred, A)
        elif fairness_metric == "spd":
            fair = statistical_parity_score(y_ground, y_pred, A)
        elif fairness_metric == "min_tpr":
            fair = min_true_positive_rate(y_ground, y_pred, A)
        elif fairness_metric == "min_pr":
            fair = min_positive_rate(y_ground, y_pred, A)
        elif fairness_metric == "min_bal_acc":
            fair = min_balanced_accuracy(y_ground, y_pred, A)
        elif fairness_metric == "min_acc":
            fair = min_accuracy(y_ground, y_pred, A)

        return alpha * perf + (1 - alpha) * (1 - abs(fair))

    return scorer


def get_fairness_goal_scorer(
    fairness_goal, M=10, performance_metric="acc", fairness_metric="eod"
):
    """Create a scorer for fairness metrics. The scorer can be used in hyperparameter tuning.

    It will return the value of the roc auc if the fairness goal is reached, otherwise it will return a low value.

    Parameters
    ----------
    fairness_goal : float
        Value of the fairness metric to be reached. The lower the value, the more fair the model.
    performance_metric : str, optional
        Performance metric in ["acc", "roc_auc"], by default "acc"
    fairness_metric : str, optional
        Fairness metric in ["eod", "spd"], by default "eod"
    M : int, optional
        Penalty for not reaching the fairness goal, by default 10
    """

    def scorer(y_ground, y_pred, A):
        if performance_metric == "acc":
            perf = accuracy_score(y_ground, y_pred)
        elif performance_metric == "bal_acc":
            perf = balanced_accuracy_score(y_ground, y_pred)
        elif performance_metric == "roc_auc":
            perf = roc_auc_score(y_ground, y_pred)

        if fairness_metric == "eod":
            fair = 1 - abs(equal_opportunity_score(y_ground, y_pred, A))
        elif fairness_metric == "spd":
            fair = 1 - abs(statistical_parity_score(y_ground, y_pred, A))

        if fair >= fairness_goal:
            return perf
        else:
            return perf - M * abs(fair - fairness_goal)

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


def logloss_group(y_ground, y_prob, A, fairness_constraint):
    """For each subgroup, calculates the mean log loss of the samples."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if fairness_constraint == "equalized_loss":
        loss = -(y_ground * np.log(y_prob) + (1 - y_ground) * np.log(1 - y_prob))
    if fairness_constraint == "demographic_parity":
        y_ = np.ones(y_ground.shape[0])  # all positive class
        loss = -(y_ground * np.log(y_prob) + (1 - y_ground) * np.log(1 - y_prob))
    elif fairness_constraint == "equal_opportunity":
        loss = -(y_ground * np.log(y_prob) + (1 - y_ground) * np.log(1 - y_prob))
        loss[y_ground == 0] = 0  # only consider the loss of the positive class

    loss = np.array([np.mean(loss[A == a]) for a in np.unique(A)])
    return loss


def max_logloss_score(y_ground, y_prob, A):
    """Calculate the minimum mean loss of the groups. The loss is binary cross entropy.
    It work with multiple groups.

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
        Minimum mean loss of groups
    """
    logloss = logloss_group(y_ground, y_prob, A, "equalized_loss")
    return max(logloss)


def logloss_score(y_ground, y_pred):
    return -np.mean(y_ground * np.log(y_pred) + (1 - y_ground) * np.log(1 - y_pred))
