import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import cvxpy as cp


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """_summary_

    :param alpha: weight of the performance objective, must be in [0, 1], 1 - alpha is the weight of the fairness objective
    :param ClassifierMixin: _description_
    """

    def __init__(
        self,
        n_estimators=10,
        eta=0.3,
        colsample_bytree=1,
        max_depth=6,
        min_child_weight=1,
        max_leaves=0,
        l2_weight=1,
        objective=None,
        alpha=1,
        sensitive_idx=None,
        seed=None,
    ):
        self.eta = eta
        self.n_estimators = n_estimators
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.l2_weight = l2_weight
        self.max_leaves = max_leaves
        self.objective = objective
        self.alpha = alpha
        self.sensitive_idx = sensitive_idx
        self.seed = seed

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "tree_method": "hist",
            "objective": "binary:logistic",
            "max_depth": self.max_depth,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "max_leaves": self.max_leaves,
            "eta": self.eta,
            "lambda": self.l2_weight,
        }
        if self.seed is not None:
            params["seed"] = self.seed

        self.model_ = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators, obj=self.objective
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds_pos = self.model_.predict(dtest)
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds

    def fairness_score(self, X, y, preds):
        A = X[:, self.sensitive_idx]
        A = A.reshape(-1)
        A = A.astype(int)
        EOD = np.mean(preds[(A == 1) & (y == 1)]) - np.mean(preds[(A == 0) & (y == 1)])
        return 1 - np.abs(EOD)

    def score(self, X, y):
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        roc = roc_auc_score(y, preds)
        if self.alpha == 1:
            return roc
        fair = self.fairness_score(X, y, preds)
        return roc * self.alpha + (1 - self.alpha) * fair


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


def logloss_grad_group(predt, dtrain, subgroup):
    """Compute the gradient for log loss for each group."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    grad = -(y - predt)

    groups = np.unique(subgroup)
    grad_matrix = np.zeros((len(y), len(groups)))

    for i, group in enumerate(groups):
        grad_matrix[:, i] = grad  # copy the column
        grad_matrix[subgroup != group, i] = 0  # and set 0 for other groups
    return grad_matrix


def logloss_hessian_group(predt, dtrain, subgroup):
    """Compute the hessian for log loss for each group."""
    predt = 1 / (1 + np.exp(-predt))
    hess = predt * (1 - predt)

    groups = np.unique(subgroup)
    hess_matrix = np.zeros((len(hess), len(groups)))

    for i, group in enumerate(groups):
        hess_matrix[:, i] = hess  # copy the column
        hess_matrix[subgroup != group, i] = 0  # and set 0 for other groups
    return hess_matrix


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


def penalize_max_loss_subgroups(subgroup_idx, fair_weight):
    weight_1 = 1
    weight_2 = fair_weight

    def custom_obj(predt, dtrain):
        subgroup = (dtrain.get_data()[:, subgroup_idx]).toarray().reshape(-1)

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

        n = len(subgroup)
        subgroup_ind = (get_subgroup_indicator(subgroup) * mu_opt).sum(axis=1)
        grad = logloss_grad(predt, dtrain) * (1 / n + subgroup_ind)
        hess = logloss_hessian(predt, dtrain) * (1 / n + subgroup_ind)

        return grad, hess

    return custom_obj

def dual_obj(subgroup, fair_weight):
    weight_1 = 1
    weight_2 = fair_weight
    n = len(subgroup)
    n_g = get_subgroup_indicator(subgroup)

    def custom_obj(predt, dtrain):
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
        
        grad = logloss_grad(predt, dtrain) * (1 / n + np.sum(n_g * mu_opt, axis=1))
        hess = logloss_hessian(predt, dtrain) * (1 / n + np.sum(n_g * mu_opt, axis=1))
        return grad, hess

    return custom_obj

class XtremeFair(BaseEstimator, ClassifierMixin):
    """Gradient Boosting with min max fairness regularization.

    :param n_estimators:
    :param ClassifierMixin: _description_
    """

    def __init__(
        self,
        fairness_constraint = "equalized_loss",
        fair_weight=1,
        use_sensitive_attr = True,
        n_estimators=10,
        eta=0.3,
        colsample_bytree=1,
        max_depth=6,
        min_child_weight=1,
        max_leaves=0,
        l2_weight=1,
        alpha=1,
        fairness_metric = "EOD",
        seed=None,
    ):  
        if fairness_constraint != "equalized_loss":
            raise NotImplementedError(f"Fairness constraint {fairness_constraint} not implemented.")
        
        if fairness_metric != "EOD":
            raise NotImplementedError(f"Fairness score {fairness_metric} not implemented.")
        
        self.fairness_constraint = fairness_constraint
        self.fair_weight = fair_weight
        self.use_sensitive_attr = use_sensitive_attr
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
        """_summary_

        :param X: array-like of shape (n_samples, n_features), sensitive attribute must be in the first column
        :param y: labels array-like of shape (n_samples), must be (0 or 1)
        :return: self
        """
        X, y = check_X_y(X, y)
        A = X[:, 0].reshape(-1)
        self.classes_ = np.unique(y)
        if not self.use_sensitive_attr:
            dtrain = xgb.DMatrix(X[:, 1:], label=y)
        else:
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
            obj=dual_obj(A, self.fair_weight)
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if not self.use_sensitive_attr:
            dtest = xgb.DMatrix(X[:, 1:])
        else:
            dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if not self.use_sensitive_attr:
            dtest = xgb.DMatrix(X[:, 1:])
        else:
            dtest = xgb.DMatrix(X)
        preds_pos = self.model_.predict(dtest)
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds

    def fairness_score(self, X, y, preds):
        A = X[:, 0].reshape(-1)
        EOD = np.mean(preds[(A == 1) & (y == 1)]) - np.mean(preds[(A == 0) & (y == 1)])
        return 1 - np.abs(EOD)

    def score(self, X, y):
        check_is_fitted(self)
        X = check_array(X)
        if not self.use_sensitive_attr:
            dtest = xgb.DMatrix(X[:, 1:])
        else:
            dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        acc = accuracy_score(y, preds > 0.5)
        if self.alpha == 1:
            return acc
        fair = self.fairness_score(X, y, preds)
        return acc * self.alpha + (1 - self.alpha) * fair
