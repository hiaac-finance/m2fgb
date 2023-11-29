import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
import xgboost as xgb


def logloss_grad(predt, dtrain):
    """Compute the gradient for cross entropy log loss."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    grad = -(y - predt)
    return grad


def logloss_hessian(predt, dtrain):
    """Compute the hessian for cross entropy log loss."""
    predt = 1 / (1 + np.exp(-predt))
    hess = predt * (1 - predt)
    return hess


def logloss_group(predt, dtrain, subgroup):
    """For each subgroup, calculates the mean log loss of the samples."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    loss = -(y * np.log(predt) + (1 - y) * np.log(1 - predt))
    groups = np.unique(subgroup)

    loss_matrix = np.zeros((len(y), len(groups)))
    for i, group in enumerate(groups):
        loss_matrix[:, i] = loss  # copy the column
        loss_matrix[subgroup != group, i] = np.nan  # and set nan for other groups

    return np.nanmean(loss_matrix, axis=0)


def get_subgroup_indicator(subgroup):
    """Return matrix with a column for each subgroup.
    Each column has value 1/n_g for the samples in the subgroup and 0 otherwise.
    """
    groups = np.unique(subgroup)
    subgroup_ind = np.zeros((len(subgroup), len(groups)))

    for i, group in enumerate(groups):
        subgroup_ind[:, i] = subgroup == group
        n_g = np.sum(subgroup_ind[:, i])
        subgroup_ind[:, i] = subgroup_ind[:, i] / n_g
    return subgroup_ind


def dual_obj(fair_weight):
    """This helper function will define a custom objective function for XGBoost
    using the fair_weight parameter.

    :param fair_weight: weight of the fairness loss term.
    :type fair_weight: float
    :return: objective function defined with the fair_weight
    :rtype: function
    """
    weight_1 = 1
    weight_2 = fair_weight

    def custom_obj(predt, dtrain):
        subgroup = (dtrain.get_data()[:, 0]).toarray().reshape(-1)
        n = len(subgroup)
        n_g = get_subgroup_indicator(subgroup)
        if weight_2 > 0:
            # dual problem solved analytically
            loss_group = logloss_group(predt, dtrain, subgroup)
            idx_biggest_loss = np.where(loss_group == np.max(loss_group))[0]
            # if is more than one, randomly choose one
            idx_biggest_loss = np.random.choice(idx_biggest_loss)
            mu_opt = np.zeros(loss_group.shape[0])
            mu_opt[idx_biggest_loss] = weight_2

        else:
            mu_opt = np.zeros(len(np.unique(subgroup)))

        multiplier = n / (1 + weight_2) * (1 / n + np.sum(n_g * mu_opt, axis=1))
        grad = logloss_grad(predt, dtrain) * multiplier
        hess = logloss_hessian(predt, dtrain) * multiplier
        return grad, hess

    return custom_obj


class XtremeFair(BaseEstimator, ClassifierMixin):
    """Classifier that modifies XGBoost to incorporate fairness into the loss function.
    The scoring is performed with a weighted sum between accuracy and a fairness metric.
    The alpha parameter controls the weight, alpha=1 means only accuracy is considered, alpha=0 means only fairness is considered.
    It shares many of the parameters with XGBoost to control learning and decision trees.
    Currently it is only able to incorporate "equalized_loss" as a fairness constraint and "EOD" as a fairness metric.

    :param fairness_constraint: type of constraint, defaults to "equalized_loss"
    :type fairness_constraint: str, optional
    :param fair_weight: weight of fairness term, defaults to 1
    :type fair_weight: float, optional
    :param n_estimators: number of trees, defaults to 10
    :type n_estimators: int, optional
    :param eta: learning rate of boosting, defaults to 0.3
    :type eta: float, optional
    :param colsample_bytree: fraction of data used in learning each tree, defaults to 1
    :type colsample_bytree: float, optional
    :param max_depth: max depth of each tree, lower values reduce complexity, defaults to 6
    :type max_depth: int, optional
    :param min_child_weight: weight to decide in node split of each tree, lower values increase complexity, defaults to 1
    :type min_child_weight: float, optional
    :param max_leaves: number of max leaves per tree, defaults to 0
    :type max_leaves: int, optional
    :param l2_weight: weight of L2 regularization of trees, defaults to 1
    :type l2_weight: float, optional
    :param alpha: weight of performance-fairness score, must be in [0, 1], defaults to 1
    :type alpha: float, optional
    :param fairness_metric: fairness metric, only supports "EOD", defaults to "EOD"
    :type fairness_metric: str, optional
    :param seed: random seed, defaults to None
    :type seed: int, optional
    """

    def __init__(
        self,
        fairness_constraint="equalized_loss",
        fair_weight=1,
        n_estimators=10,
        eta=0.3,
        colsample_bytree=1,
        max_depth=6,
        min_child_weight=1,
        max_leaves=0,
        l2_weight=1,
        alpha=1,
        fairness_metric="EOD",
        seed=None,
    ):
        if fairness_constraint != "equalized_loss":
            raise NotImplementedError(
                f"Fairness constraint {fairness_constraint} not implemented."
            )

        if fairness_metric != "EOD":
            raise NotImplementedError(
                f"Fairness score {fairness_metric} not implemented."
            )

        self.fairness_constraint = fairness_constraint
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.eta = eta
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_leaves = max_leaves
        self.l2_weight = l2_weight
        self.alpha = alpha
        self.seed = seed
        self.fairness_metric = fairness_metric

    def fit(self, X, y):
        """Fit the model to the data.

        :param X: dataframe of shape (n_samples, n_features), sensitive attribute must be in the first column
        :type X: pandas.DataFrame
        :param y: labels array-like of shape (n_samples), must be (0 or 1)
        :type y: pandas.Series
        :return: fitted model
        :rtype: XtremeFair
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eta": self.eta,
            "colsample_bytree": self.colsample_bytree,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "max_leaves": self.max_leaves,
            "lambda": self.l2_weight,
        }
        if self.seed is not None:
            params["seed"] = self.seed

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=dual_obj(self.fair_weight),
        )
        return self

    def predict(self, X):
        """Predict the labels of the data."""
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict the probabilities of the data."""
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds_pos = self.model_.predict(dtest)
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds

    def fairness_score(self, X, y, preds):
        """Calculate the fairness score of the model."""
        A = X[:, 0].reshape(-1)
        EOD = np.mean(preds[(A == 1) & (y == 1)]) - np.mean(preds[(A == 0) & (y == 1)])
        return 1 - np.abs(EOD)

    def score(self, X, y):
        """Calculate the performance-fairness score of the model."""
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        acc = accuracy_score(y, preds > 0.5)
        if self.alpha == 1:
            return acc
        fair = self.fairness_score(X, y, preds)
        return acc * self.alpha + (1 - self.alpha) * fair
