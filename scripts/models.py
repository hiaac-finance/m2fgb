import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, log_loss
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    TruePositiveRateParity,
    EqualizedOdds,
)
from sklego.linear_model import DemographicParityClassifier, EqualOpportunityClassifier
from sklearn.linear_model import LogisticRegression

import utils
import lightgbm as lgb
import fairgbm

import sys
sys.path.append("../minimax-fair/src")
import minmaxML as mmml

lgb.register_logger(utils.CustomLogger())
fairgbm.register_logger(utils.CustomLogger())

PARAM_SPACES = {
    "M2FGB_XGB": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 1000, "log": True},
        "eta": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "l2_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 0.001, "high": 10, "log": True},
    },
    "M2FGB_XGB_grad": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 1000, "log": True},
        "eta": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "l2_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 0.001, "high": 10, "log": True},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.5,
            "log": True,
        },
    },
    "M2FGB": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 10, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 7},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 0.001, "high": 10, "log": True},
    },
    "M2FGB_grad": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 10, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 7},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 1e-2, "high": 1},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.5,
            "log": True,
        },
    },
    "FairGBMClassifier": {
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "min_child_samples": {"type": "int", "low": 5, "high": 250, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 7},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.5,
            "log": True,
        },
        "constraint_fnr_tolerance": {
            "type": "float",
            "low": 0.005,
            "high": 0.5,
            "log": True,
        },
    },
    "LGBMClassifier": {
        "num_leaves": {"type": "int", "low": 2, "high": 1000, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "min_child_samples": {"type": "int", "low": 5, "high": 500, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
    },
    "ExponentiatedGradient": {
        "eps": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
        "max_iter": {"type": "int", "low": 10, "high": 1000, "log": True},
        "eta0": {"type": "float", "low": 0.1, "high": 100, "log": True},
        "min_child_leaf": {"type": "int", "low": 5, "high": 500, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "criterion": {"type": "str", "options": ["gini", "entropy"]},
    },
    "FairClassifier": {
        "covariance_threshold": {"type": "float", "low": 0.1, "high": 1, "log": True},
        "C": {"type": "float", "low": 0.1, "high": 1000, "log": True},
        "penalty": {"type": "str", "options": ["none", "l1"]},
    },
    "MinMaxFair": {
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "gamma": {"type": "float", "low": 0, "high": 1},
        "penalty": {"type": "str", "options": ["none", "l1"]},
        "C": {"type": "float", "low": 0.1, "high": 1000, "log": True},
        "a": {"type": "float", "low": 0.1, "high": 1},
        "b": {"type": "float", "low": 1e-2, "high": 1},
    },
}


def projection_to_simplex(mu, z=1):
    sorted_mu = mu[np.argsort(mu)]
    n = len(mu)
    t = np.mean(mu) - z / n
    for i in range(len(mu) - 2, -1, -1):
        t_i = np.mean(sorted_mu[(i + 1) :]) - z / (n - i - 1)
        if t_i >= sorted_mu[i]:
            t = t_i
            break

    x = mu - t
    x = np.where(x > 0, x, 0)
    return x


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


def logloss_group(predt, dtrain, subgroup, fairness_constraint):
    """For each subgroup, calculates the mean log loss of the samples."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    predt = np.clip(predt, 1e-7, 1 - 1e-7)  # avoid log(0)
    if fairness_constraint == "equalized_loss":
        loss = -(y * np.log(predt) + (1 - y) * np.log(1 - predt))
    if fairness_constraint == "demographic_parity":
        y_ = np.ones(y.shape[0])  # all positive class
        loss = -(y_ * np.log(predt) + (1 - y_) * np.log(1 - predt))
    elif fairness_constraint == "equal_opportunity":
        loss = -(y * np.log(predt) + (1 - y) * np.log(1 - predt))
        loss[y == 0] = 0  # only consider the loss of the positive class

    loss = np.array([np.mean(loss[subgroup == g]) for g in np.unique(subgroup)])
    return loss


def logloss_group_grad(predt, dtrain, fairness_constraint):
    """Create an array with the gradient of fairness metrics."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    if fairness_constraint == "equalized_loss":
        grad = -(y - predt)
    elif fairness_constraint == "demographic_parity":
        y_ = np.ones(y.shape[0])  # all positive class
        grad = -(y_ - predt)
    elif fairness_constraint == "equal_opportunity":
        grad = -(y - predt)
        grad[y == 0] = 0  # only consider the loss of the positive class

    return grad


def logloss_group_hess(predt, dtrain, fairness_constraint):
    """Create an array with the hessian of fairness metrics."""
    y = dtrain.get_label()
    predt = 1 / (1 + np.exp(-predt))
    if fairness_constraint == "equalized_loss":
        hess = predt * (1 - predt)
    elif (
        fairness_constraint == "demographic_parity"
        or fairness_constraint == "equal_opportunity"
    ):
        hess = predt * (1 - predt)
        hess[y == 0] = 0  # only consider the loss of the positive class

    return hess


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


def dual_obj(
    subgroup,
    fair_weight,
    group_losses,
    fairness_constraint="equalized_loss",
    dual_learning="optim",
    multiplier_learning_rate=0.1,
):
    """This helper function will define a custom objective function for XGBoost using the fair_weight parameter.

    Parameters
    ----------
    soubgroup : ndarray
        Array with the subgroup labels.
    fair_weight : float
        Weight of the fairness term in the loss function.
    group_losses : list
        List where the losses for each subgroup will be stored.
    fairness_constraint: str, optional
        Fairness constraint used in learning.
    dual_learning : str, optional
        Method used to learn the dual problem, by default "optim"
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, used only if dual_learning="gradient", by default 0.1
    """
    mu_opt_list = [None]
    n = len(subgroup)
    n_g = get_subgroup_indicator(subgroup)

    def custom_obj(predt, dtrain):
        loss_group = logloss_group(predt, dtrain, subgroup, fairness_constraint)
        group_losses.append(loss_group)
        if fair_weight > 0:
            if dual_learning == "optim":
                # dual problem solved analytically
                idx_biggest_loss = np.where(loss_group == np.max(loss_group))[0]
                # if is more than one, randomly choose one
                idx_biggest_loss = np.random.choice(idx_biggest_loss)
                mu_opt = np.zeros(loss_group.shape[0])
                mu_opt[idx_biggest_loss] = fair_weight
                if mu_opt_list[0] is None:
                    mu_opt_list[0] = mu_opt
                else:
                    mu_opt_list.append(mu_opt)

            elif dual_learning == "gradient":
                if mu_opt_list[0] is None:
                    mu_opt = np.zeros(loss_group.shape[0])
                    # mu_opt = mu_opt / np.sum(mu_opt) * fair_weight
                    mu_opt_list[0] = mu_opt

                else:
                    mu_opt = mu_opt_list[-1]
                    mu_opt += multiplier_learning_rate * fair_weight * loss_group
                    mu_opt_list.append(mu_opt)

        else:
            mu_opt = np.zeros(len(np.unique(subgroup)))

        grad = logloss_grad(predt, dtrain) / n
        hess = logloss_hessian(predt, dtrain) / n
        if fair_weight > 0:
            grad += (
                logloss_group_grad(predt, dtrain, subgroup, fairness_constraint)
                * n_g
                @ mu_opt
            )
            hess += (
                logloss_group_hess(predt, dtrain, subgroup, fairness_constraint)
                * n_g
                @ mu_opt
            )

        grad *= n / (1 + fair_weight)
        hess *= n / (1 + fair_weight)
        return grad, hess

    return custom_obj


class M2FGB_XGB(BaseEstimator, ClassifierMixin):
    """Classifier that modifies XGBoost to incorporate fairness into the loss function.
    It shares many of the parameters with XGBoost to control learning and decision trees.
    The fairness metrics impelemented are "equalized_loss", "equal_opportunity", and "demographic_parity".


    Parameters
    ----------
    fairness_constraint : str, optional
        Fairness constraint used in learning, currently only supports "equalized_loss", by default "equalized_loss"
    fair_weight : int, optional
        Weight for fairness in loss formulation, by default 1
    dual_learning : str, optional
        Method used to learn the dual problem, by default "optim"
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, used only if dual_learning="gradient", by default 0.1
    n_estimators : int, optional
        Number of estimators used in XGB, by default 10
    eta : float, optional
        Learning rate of XGB, by default 0.3
    colsample_bytree : float, optional
        Size of sample of of the columns used in each estimator, by default 1
    max_depth : int, optional
        Max depth of decision trees of XGB, by default 6
    min_child_weight : int, optional
        Weight used to choose partition of tree nodes, by default 1
    max_leaves : int, optional
        Max number of leaves of trees, by default 0
    l2_weight : int, optional
        Weight of L2 regularization, by default 1
    random_state : int, optional
        Random seed used in learning, by default None
    """

    def __init__(
        self,
        fairness_constraint="equalized_loss",
        fair_weight=1,
        dual_learning="optim",
        multiplier_learning_rate=0.1,
        n_estimators=10,
        eta=0.3,
        colsample_bytree=1,
        max_depth=6,
        min_child_weight=1,
        max_leaves=0,
        l2_weight=1,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "equal_opportunity",
            "demographic_parity",
        ]
        assert dual_learning in ["optim", "gradient"]

        self.fairness_constraint = fairness_constraint
        self.dual_learning = dual_learning
        self.multiplier_learning_rate = multiplier_learning_rate
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.eta = eta
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_leaves = max_leaves
        self.l2_weight = l2_weight
        self.random_state = random_state
        self.group_losses = []

    def fit(self, X, y, sensitive_attribute=None):
        """Fit the model to the data.


        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe of shape (n_samples, n_features)
        y : pandas.Series or numpy.ndarray
            Labels array-like of shape (n_samples), must be (0 or 1)
        sensitive_attribute : pandas.Series or numpy.ndarray
            Sensitive attribute array-like of shape (n_samples)

        Returns
        -------
        M2FGB
            Fitted model
        """
        if sensitive_attribute is None:
            sensitive_attribute = np.ones(X.shape[0])
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
        if self.random_state is not None:
            params["seed"] = self.random_state

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=dual_obj(
                sensitive_attribute,
                self.fair_weight,
                self.group_losses,
                self.fairness_constraint,
                self.dual_learning,
                self.multiplier_learning_rate,
            ),
        )
        self.group_losses = np.array(self.group_losses)
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


def get_subgroup_indicator_test(subgroup):
    groups = np.unique(subgroup)
    n = len(subgroup)
    I = np.zeros((subgroup.shape[0], len(groups)))
    n_g_max = -np.inf
    for i, g in enumerate(groups):
        n_g = np.sum(subgroup == g)
        n_g_max = max(n_g_max, n_g)
        I[subgroup == g, i] = 1 / np.sum(subgroup == g)

    I = I * n
    return I


def dual_obj_1(
    subgroup,
    fair_weight,
    group_losses,
    mu_opt_list,
    fairness_constraint="equalized_loss",
    dual_learning="optim",
    multiplier_learning_rate=0.1,
):
    """This helper function will define a custom objective function for XGBoost using the fair_weight parameter.

    Parameters
    ----------
    soubgroup : ndarray
        Array with the subgroup labels.
    fair_weight : float
        Weight of the fairness term in the loss function.
    group_losses : list
        List where the losses for each subgroup will be stored.
    mu_opt_list: list
        List where the optimal mu for each subgroup will be stored.
    fairness_constraint: str, optional
        Fairness constraint used in learning.
    dual_learning : str, optional
        Method used to learn the dual problem, by default "optim"
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, used only if dual_learning="gradient", by default 0.1
    """
    I = get_subgroup_indicator_test(subgroup)

    def custom_obj(predt, dtrain):
        loss_group = logloss_group(predt, dtrain, subgroup, fairness_constraint)
        group_losses.append(loss_group)

        if dual_learning == "optim":
            # dual problem solved analytically
            idx_biggest_loss = np.where(loss_group == np.max(loss_group))[0]
            # if is more than one, randomly choose one
            idx_biggest_loss = np.random.choice(idx_biggest_loss)
            mu_opt = np.zeros(loss_group.shape[0])
            mu_opt[idx_biggest_loss] = fair_weight

        elif dual_learning == "gradient":
            if mu_opt_list[0] is None:
                mu_opt = np.zeros(loss_group.shape[0])
            else:
                mu_opt = mu_opt_list[-1].copy()
            mu_opt += multiplier_learning_rate * fair_weight * loss_group

        elif dual_learning == "gradient_norm":
            if mu_opt_list[0] is None:
                mu_opt = np.ones(loss_group.shape[0])
            else:
                mu_opt = mu_opt_list[-1].copy()

            mu_opt += multiplier_learning_rate * fair_weight * loss_group
            mu_opt = projection_to_simplex(mu_opt, z=fair_weight)

        if mu_opt_list[0] is None:
            mu_opt_list[0] = mu_opt
        else:
            mu_opt_list.append(mu_opt)

        grad_fair = logloss_group_grad(predt, dtrain, fairness_constraint)
        grad_fair = I * grad_fair.reshape(-1, 1) @ mu_opt

        hess_fair = logloss_group_hess(predt, dtrain, fairness_constraint)
        hess_fair = I * hess_fair.reshape(-1, 1) @ mu_opt

        grad = logloss_grad(predt, dtrain)
        hess = logloss_hessian(predt, dtrain)

        # It is not necessary to multiply fairness gradient by fair_weight because it is already included on mu
        # grad = (1 - fair_weight) * grad + fair_weight * grad_fair
        # hess = (1 - fair_weight) * hess + fair_weight * hess_fair

        grad = (1 - fair_weight) * grad + grad_fair
        hess = (1 - fair_weight) * hess + hess_fair

        return grad, hess

    return custom_obj


class M2FGB(BaseEstimator, ClassifierMixin):
    """Classifier that modifies LGBM to incorporate min-max fairness optimization.
    It shares many of the parameters with LGBM to control learning and decision trees.
    The fairness metrics impelemented are "equalized_loss", "equal_opportunity", and "demographic_parity".

    Parameters
    ----------
    fairness_constraint : str, optional
        Fairness constraint used in learning, must be ["equalized_loss", "equal_opportunity", "demographic_parity"], by default "equalized_loss"
    fair_weight : int, optional
        Weight for fairness in loss formulation, by default 1
    dual_learning : str, optional
        Method used to learn the dual problem, must be ["optim", "grad"], by default "optim"
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, used only if dual_learning="gradient", by default 0.1
    n_estimators : int, optional
        Number of estimators used in XGB, by default 10
    learning_rate : float, optional
        Learning rate of ensambles, by default 0.1
    max_depth : int, optional
        Max depth of decision trees, by default 6
    min_child_weight : int, optional
        Weight used to choose partition of tree nodes, by default 1
    reg_lambda : int, optional
        Weight of L2 regularization, by default 1
    random_state : int, optional
        Random seed used in learning, by default None
    """

    def __init__(
        self,
        fairness_constraint="equalized_loss",
        fair_weight=0.5,
        dual_learning="optim",
        multiplier_learning_rate=0.1,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        reg_lambda=1,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "equal_opportunity",
            "demographic_parity",
        ]
        assert dual_learning in ["optim", "gradient", "gradient_norm"]

        assert fair_weight >= 0 and fair_weight <= 1

        self.fairness_constraint = fairness_constraint
        self.dual_learning = dual_learning
        self.multiplier_learning_rate = multiplier_learning_rate
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y, sensitive_attribute):
        """Fit the model to the data.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe of shape (n_samples, n_features)
        y : pandas.Series or numpy.ndarray
            Labels array-like of shape (n_samples), must be (0 or 1)
        sensitive_attribute : pandas.Series or numpy.ndarray
            Sensitive attribute array-like of shape (n_samples)

        Returns
        -------
        M2FGB
            Fitted model
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.group_losses = []
        self.mu_opt_list = [None]
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": dual_obj_1(
                sensitive_attribute,
                self.fair_weight,
                self.group_losses,
                self.mu_opt_list,
                self.fairness_constraint,
                self.dual_learning,
                self.multiplier_learning_rate,
            ),
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "verbose": -1,
        }
        if self.random_state is not None:
            params["random_seed"] = self.random_state

        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
        self.group_losses = np.array(self.group_losses)
        self.mu_opt_list = np.array(self.mu_opt_list)
        return self

    def predict(self, X):
        """Predict the labels of the data."""
        check_is_fitted(self)
        X = check_array(X)
        log_odds = self.model_.predict(X)
        preds = 1 / (1 + np.exp(-log_odds))
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict the probabilities of the data."""
        check_is_fitted(self)
        X = check_array(X)
        log_odds = self.model_.predict(X)
        preds_pos = 1 / (1 + np.exp(-log_odds))
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds


class ExponentiatedGradient_Wrap(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        fairness_constraint,
        eps,
        max_iter,
        eta0,
        min_samples_leaf,
        max_depth,
        criterion,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "equal_opportunity",
            "demographic_parity",
        ]
        fairness_mapper = {
            "equalized_loss": EqualizedOdds(),
            "equal_opportunity": TruePositiveRateParity(),
            "demographic_parity": DemographicParity(),
        }
        self.fairness_constraint = fairness_mapper[fairness_constraint]
        self.eps = eps
        self.max_iter = max_iter
        self.eta0 = eta0
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion

        self.estimator = DecisionTreeClassifier(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=random_state,
        )

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.model_ = ExponentiatedGradient(
            self.estimator,
            constraints=self.fairness_constraint,
            eps=self.eps,
            max_iter=self.max_iter,
            eta0=self.eta0,
        )
        self.model_.fit(X, y, sensitive_features=sensitive_attribute)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict_proba(X)


class FairClassifier_Wrap(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        fairness_constraint="equal_opportunity",
        covariance_threshold=0.1,
        C=1.0,
        penalty="l1",
        max_iter=100,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equal_opportunity",
            "demographic_parity",
        ]
        self.fairness_constraint = fairness_constraint
        self.covariance_threshold = covariance_threshold
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        # insert sensitive attribute as first column of the dataframe
        X = X.copy()
        X = np.insert(X, 0, sensitive_attribute, axis=1)
        self.classes_ = np.unique(y)
        if self.fairness_constraint == "equal_opportunity":
            self.model_ = EqualOpportunityClassifier(
                covariance_threshold=self.covariance_threshold,
                positive_target=1,
                C=self.C,
                sensitive_cols=0,
                penalty=self.penalty,
                max_iter=self.max_iter,
                train_sensitive_cols=False,
            )
        elif self.fairness_constraint == "demographic_parity":
            self.model_ = DemographicParityClassifier(
                covariance_threshold=self.covariance_threshold,
                C=self.C,
                sensitive_cols=0,
                penalty=self.penalty,
                max_iter=self.max_iter,
                train_sensitive_cols=False,
            )

        try:
            self.model_.fit(X, y)
        except:
            print("Error in FairClassifier")
            self.model_ = LogisticRegression(
                C=self.C,
                penalty=self.penalty,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="saga",
            )
            self.model_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = X.copy()
        A = np.ones(X.shape[0])
        X = np.insert(X, 0, A, axis=1)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = X.copy()
        A = np.ones(X.shape[0])
        X = np.insert(X, 0, A, axis=1)
        return self.model_.predict_proba(X)


class LGBMClassifier(lgb.LGBMClassifier):
    def fit(self, X, Y, A):
        super().fit(X, Y)
        return self


class FairGBMClassifier(fairgbm.FairGBMClassifier):
    def fit(self, X, Y, A):
        super().fit(X, Y, constraint_group=A)
        return self


class MinMaxFair(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        a=1,
        b=0.5,
        gamma=0.0,
        relaxed=False,
        penalty=None,
        C=1.0,
    ):
        self.n_estimators = n_estimators
        self.a = a
        self.b = b
        self.gamma = gamma
        self.relaxed = relaxed
        self.penalty = penalty
        self.C = C

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # train a logistic model to get min and max logloss
        model = LogisticRegression(
            penalty=self.penalty,
            C=1 if self.penalty == "none" else self.C,
            solver="saga",
        )
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]
        min_logloss = np.inf
        max_logloss = -np.inf
        for g in np.unique(sensitive_attribute):
            idx = sensitive_attribute == g
            logloss = log_loss(y[idx], y_pred[idx])
            min_logloss = min(min_logloss, logloss)
            max_logloss = max(max_logloss, logloss)

        gamma_hat = min_logloss + self.gamma * (max_logloss - min_logloss)

        (
            group_error,
            high_gamma,
            first_poperr,
            agg_grouperrs,
            agg_poperrs,
            _,
            pop_error_type,
            total_steps,
            modelhats,
            _,
            _,
            _,
            _,
            _,
        ) = mmml.do_learning(
            X=X,
            y=y,
            grouplabels=sensitive_attribute,
            group_names=(),
            # minmax fairness parameters
            numsteps=self.n_estimators,
            a=self.a,
            b=self.b,
            convergence_threshold=1e-15,
            scale_eta_by_label_range=True,
            equal_error=False,
            gamma=gamma_hat,
            relaxed=True,
            error_type="Log-Loss",
            extra_error_types=set(),
            pop_error_type="Log-Loss",
            # data transform
            rescale_features=False,
            test_size=0.0,
            random_split_seed=0,
            # model selection
            model_type="LogisticRegression",
            # parameters related to LR model
            max_logi_iters=1000,
            tol=1e-8,
            fit_intercept=True,
            logistic_solver="lbfgs",
            penalty=self.penalty,
            C=self.C,
            # parameters related to MLP
            lr=0.01,
            momentum=0.9,
            weight_decay=0,
            n_epochs=10000,
            hidden_sizes=(2 / 3,),
            # parameters related to output
            display_plots=False,
            verbose=False,
            use_input_commands=False,
            show_legend=False,
            save_models=False,
            save_plots=False,
            dirname="",
            data_name="",
            # not used
            group_types=(),
        )

        self.model = modelhats
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        predictions = self.predict_proba(X)[:, 1]
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # random select a model for each line of X
        predictions = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            predictions[i] = self.model[np.random.choice(len(self.model))].predict_proba(
                X[i].reshape(1, -1)
            )
        return predictions
