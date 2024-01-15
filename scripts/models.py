import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    TruePositiveRateParity,
    EqualizedOdds,
)
from sklego.linear_model import DemographicParityClassifier, EqualOpportunityClassifier
from sklearn.linear_model import LogisticRegression

PARAM_SPACES = {
    "XtremeFair": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "eta": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "l2_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 0.01, "high": 10, "log": True},
    },
    "XtremeFair_grad": {
        "min_child_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "eta": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "l2_weight": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 0.01, "high": 10, "log": True},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.5,
            "log": True,
        },
    },
    "FairGBMClassifier": {
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "min_child_samples": {"type": "int", "low": 5, "high": 500, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 10},
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
        "num_leaves": {"type": "int", "low": 2, "high": 500, "log": True},
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
    "FairClassifier" : {
        "covariance_threshold": {"type": "float", "low": 0.1, "high": 1, "log": True},
        "C": {"type": "float", "low": 0.1, "high": 1000, "log": True},
        "penalty" : {"type": "str", "options": ["none", "l1"]},

    }
}


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

    groups = np.unique(subgroup)

    loss_matrix = np.zeros((len(y), len(groups)))
    for i, group in enumerate(groups):
        loss_matrix[:, i] = loss  # copy the column
        loss_matrix[subgroup != group, i] = np.nan  # and set nan for other groups

    return np.nanmean(loss_matrix, axis=0)


def logloss_group_grad(predt, dtrain, subgroup, fairness_constraint):
    """Create a matrix with the gradient for each subgroup in each column."""
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

    groups = np.unique(subgroup)
    grad_matrix = np.zeros((len(y), len(groups)))
    for i, group in enumerate(groups):
        grad_matrix[:, i] = grad  # copy the column
        grad_matrix[subgroup != group, i] = 0
    return grad_matrix


def logloss_group_hess(predt, dtrain, subgroup, fairness_constraint):
    """Create a matrix with the hessian for each subgroup in each column."""
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

    groups = np.unique(subgroup)
    hess_matrix = np.zeros((len(y), len(groups)))
    for i, group in enumerate(groups):
        hess_matrix[:, i] = hess  # copy the column
        hess_matrix[subgroup != group, i] = 0
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


class XtremeFair(BaseEstimator, ClassifierMixin):
    """Classifier that modifies XGBoost to incorporate fairness into the loss function.
    The scoring is performed with a weighted sum between accuracy and a fairness metric.
    The alpha parameter controls the weight, alpha=1 means only accuracy is considered, alpha=0 means only fairness is considered.
    It shares many of the parameters with XGBoost to control learning and decision trees.
    Currently it is only able to incorporate "equalized_loss" as a fairness constraint and "EOD" as a fairness metric.


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
    alpha : int, optional
        Weight used for the score function of the method, by default 1
    performance_metric : str, optional
        Performance metric used by model score, supports ["accuracy", "auc"], by default "accuracy"
    fairness_metric : str, optional
        Fairness metric used by model score, supports ["SPD", "EOP"] by default "EOP"
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
        alpha=1,
        performance_metric="accuracy",
        fairness_metric="EOP",
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "equal_opportunity",
            "demographic_parity",
        ]
        assert dual_learning in ["optim", "gradient"]
        assert performance_metric in ["accuracy", "auc"]
        assert fairness_metric in ["EOP", "SPD"]

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
        self.alpha = alpha
        self.performance_metric = performance_metric
        self.fairness_metric = fairness_metric
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
        XtremeFair
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

    def fairness_score(self, X, y, preds):
        """Calculate the fairness score of the model. The sensitive attribute must be in the first column of X."""
        A = X[:, 0].reshape(-1)
        if self.fairness_metric == "EOP":
            fair = np.mean(preds[(A == 1) & (y == 1)]) - np.mean(
                preds[(A == 0) & (y == 1)]
            )
        elif self.fairness_metric == "SPD":
            fair = np.mean(preds[A == 1]) - np.mean(preds[A == 0])
        return 1 - np.abs(fair)

    def score(self, X, y, return_type="combined"):
        """
        Calculate the performance-fairness score of the model. The score has the following formula:

        $score = $perf * \alpha + (1 - |fair|) * (1 - \alpha)$$

        where perf is the performance metric (accuracy or AUC), fair is the fairness metric (EOD or SPD), and alpha is the weight of the performance metric.
        """
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        if self.performance_metric == "auc":
            perf = roc_auc_score(y, preds)
        elif self.performance_metric == "accuracy":
            ts = ks_threshold(y, preds)
            perf = accuracy_score(y, preds > ts)
        if self.alpha == 1:
            return perf
        fair = self.fairness_score(X, y, preds)
        combined_score = perf * self.alpha + (1 - self.alpha) * fair
        if return_type == "combined":
            return combined_score
        elif return_type == "performance":
            return perf
        elif return_type == "fairness":
            return fair
        else:
            raise ValueError(
                "Invalid return_type. Choose 'combined', 'performance', or 'fairness'."
            )


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
                    mu_opt_list[0] = mu_opt

                else:
                    mu_opt = mu_opt_list[-1].copy()
                    mu_opt += multiplier_learning_rate * fair_weight * loss_group
                    mu_opt_list.append(mu_opt)

        else:
            mu_opt = np.zeros(len(np.unique(subgroup)))
            if mu_opt_list[0] is None:
                mu_opt_list[0] = mu_opt
            else:
                mu_opt_list.append(mu_opt)

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


class XtremeFair_1(BaseEstimator, ClassifierMixin):
    """Classifier that modifies XGBoost to incorporate fairness into the loss function.
    The scoring is performed with a weighted sum between accuracy and a fairness metric.
    The alpha parameter controls the weight, alpha=1 means only accuracy is considered, alpha=0 means only fairness is considered.
    It shares many of the parameters with XGBoost to control learning and decision trees.
    Currently it is only able to incorporate "equalized_loss" as a fairness constraint and "EOD" as a fairness metric.


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
    alpha : int, optional
        Weight used for the score function of the method, by default 1
    performance_metric : str, optional
        Performance metric used by model score, supports ["accuracy", "auc"], by default "accuracy"
    fairness_metric : str, optional
        Fairness metric used by model score, supports ["SPD", "EOP"] by default "EOP"
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
        alpha=1,
        performance_metric="accuracy",
        fairness_metric="EOP",
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "equal_opportunity",
            "demographic_parity",
        ]
        assert dual_learning in ["optim", "gradient"]
        assert performance_metric in ["accuracy", "auc"]
        assert fairness_metric in ["EOP", "SPD"]

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
        self.alpha = alpha
        self.performance_metric = performance_metric
        self.fairness_metric = fairness_metric
        self.random_state = random_state
        self.group_losses = []
        self.mu_opt_list = [None]

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
        XtremeFair
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
            obj=dual_obj_1(
                sensitive_attribute,
                self.fair_weight,
                self.group_losses,
                self.mu_opt_list,
                self.fairness_constraint,
                self.dual_learning,
                self.multiplier_learning_rate,
            ),
        )
        self.group_losses = np.array(self.group_losses)
        self.mu_opt_list = np.array(self.mu_opt_list)
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
        """Calculate the fairness score of the model. The sensitive attribute must be in the first column of X."""
        A = X[:, 0].reshape(-1)
        if self.fairness_metric == "EOP":
            fair = np.mean(preds[(A == 1) & (y == 1)]) - np.mean(
                preds[(A == 0) & (y == 1)]
            )
        elif self.fairness_metric == "SPD":
            fair = np.mean(preds[A == 1]) - np.mean(preds[A == 0])
        return 1 - np.abs(fair)

    def score(self, X, y, return_type="combined"):
        """
        Calculate the performance-fairness score of the model. The score has the following formula:

        $score = $perf * \alpha + (1 - |fair|) * (1 - \alpha)$$

        where perf is the performance metric (accuracy or AUC), fair is the fairness metric (EOD or SPD), and alpha is the weight of the performance metric.
        """
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        if self.performance_metric == "auc":
            perf = roc_auc_score(y, preds)
        elif self.performance_metric == "accuracy":
            ts = ks_threshold(y, preds)
            perf = accuracy_score(y, preds > ts)
        if self.alpha == 1:
            return perf
        fair = self.fairness_score(X, y, preds)
        combined_score = perf * self.alpha + (1 - self.alpha) * fair
        if return_type == "combined":
            return combined_score
        elif return_type == "performance":
            return perf
        elif return_type == "fairness":
            return fair
        else:
            raise ValueError(
                "Invalid return_type. Choose 'combined', 'performance', or 'fairness'."
            )


def ks_threshold(y_true, y_score):
    """Identify the threshold that maximizes the Kolmogorov-Smirnov statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    return opt_threshold


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
        fairness_constraint = "equal_opportunity",
        covariance_threshold = 0.1,
        C = 1.0,
        penalty = "l1",
        max_iter = 100,
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
        X = np.insert(X, 0, sensitive_attribute, axis = 1)
        self.classes_ = np.unique(y)
        if self.fairness_constraint == "equal_opportunity":
            self.model_ = EqualOpportunityClassifier(
                covariance_threshold=self.covariance_threshold,
                positive_target = 1,
                C=self.C,
                sensitive_cols = 0,
                penalty=self.penalty,
                max_iter=self.max_iter,
                train_sensitive_cols = False,
            )
        elif self.fairness_constraint == "demographic_parity":
            self.model_ = DemographicParityClassifier(
                covariance_threshold=self.covariance_threshold,
                positive_target = 1,
                C=self.C,
                sensitive_cols = 0,
                penalty=self.penalty,
                max_iter=self.max_iter,
                train_sensitive_cols = False,
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
                solver = "saga"
            )
            self.model_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = X.copy()
        A = np.ones(X.shape[0])
        X = np.insert(X, 0, A, axis = 1)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = X.copy()
        A = np.ones(X.shape[0])
        X = np.insert(X, 0, A, axis = 1)
        return self.model_.predict_proba(X)
    
    
