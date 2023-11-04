import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import cvxpy as cp


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
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

    def score(self, X, y):
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        return roc_auc_score(y, preds)
        p = get_best_threshold(y, preds)
        return accuracy_score(y, preds > p)


def logloss_grad(predt, dtrain):
    """Compute the gradient for log loss."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    grad = -(y - predt)
    return grad


def logloss_hessian(predt, dtrain):
    """Compute the hessian for log loss."""
    predt = 1 / (1 + np.exp(-predt))
    hess = predt * (1 - predt)
    return hess


def logloss_group(predt, dtrain, subgroup):
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    loss = -(y * np.log(predt) + (1 - y) * np.log(1 - predt))
    groups = np.unique(subgroup)
    loss_matrix = np.zeros((len(y), len(groups)))

    for i, group in enumerate(groups):
        loss_matrix[:, i] = loss  # copy the column
        loss_matrix[subgroup != group, i] = 0  # and set 0 for other groups

    loss_matrix = np.sum(loss_matrix, axis=0) / np.sum(loss_matrix != 0, axis=0)
    return loss_matrix


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


def penalize_max_loss_subgroups(subgroup_idx, fair_weight):
    weight_1 = 1
    weight_2 = fair_weight

    def custom_obj(predt, dtrain):
        subgroup = (dtrain.get_data()[:, subgroup_idx]).toarray().reshape(-1)

        if weight_2 > 0:
            # dual problem
            loss_group = logloss_group(predt, dtrain, subgroup)
            mu = cp.Variable(loss_group.shape[0])  # number of groups
            z = cp.Variable(1)  # z is the min of mu * loss
            constraints = [cp.sum(mu) == weight_2, mu >= 0] + [
                z <= mu[i] * loss_group[i] for i in range(loss_group.shape[0])
            ]

            objective = cp.Maximize(z)
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # primal problem
            mu_opt = mu.value
        else:
            mu_opt = np.zeros(len(np.unique(subgroup)))

        grad_group = logloss_grad_group(predt, dtrain, subgroup)
        hess_group = logloss_hessian_group(predt, dtrain, subgroup)
        grad = logloss_grad(predt, dtrain) * weight_1 + np.sum(
            mu_opt * grad_group, axis=1
        )
        hess = logloss_hessian(predt, dtrain) * weight_1 + np.sum(
            mu_opt * hess_group, axis=1
        )

        return grad, hess

    return custom_obj
