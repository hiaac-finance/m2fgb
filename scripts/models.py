import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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
    predt = np.clip(predt, 1e-6, 1 - 1e-6) # avoid log(0)
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


def dual_obj(fair_weight, group_losses):
    """This helper function will define a custom objective function for XGBoost using the fair_weight parameter.

    Parameters
    ----------
    fair_weight : float
        Weight of the fairness term in the loss function.
    """
    def custom_obj(predt, dtrain):
        subgroup = (dtrain.get_data()[:, 0]).toarray().reshape(-1)
        n = len(subgroup)
        n_g = get_subgroup_indicator(subgroup)
        loss_group = logloss_group(predt, dtrain, subgroup)
        group_losses.append(loss_group)
        if fair_weight > 0:
            # dual problem solved analytically
            idx_biggest_loss = np.where(loss_group == np.max(loss_group))[0]
            # if is more than one, randomly choose one
            idx_biggest_loss = np.random.choice(idx_biggest_loss)
            mu_opt = np.zeros(loss_group.shape[0])
            mu_opt[idx_biggest_loss] = fair_weight

        else:
            mu_opt = np.zeros(len(np.unique(subgroup)))

        multiplier = n / (1 + fair_weight) * (1 / n + np.sum(n_g * mu_opt, axis=1))
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


    Parameters
    ----------
    fairness_constraint : str, optional
        Fairness constraint used in learning, currently only supports "equalized_loss", by default "equalized_loss"
    fair_weight : int, optional
        Weight for fairness in loss formulation, by default 1
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
    seed : int, optional
        Random seed used in learning, by default None
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
        performance_metric="accuracy",
        fairness_metric="EOP",
        seed=None,
    ):
        assert fairness_constraint in ["equalized_loss"]
        assert performance_metric in ["accuracy", "auc"]
        assert fairness_metric in ["EOP", "SPD"]

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
        self.performance_metric = performance_metric
        self.fairness_metric = fairness_metric
        self.seed = seed
        self.group_losses = []

    def fit(self, X, y):
        """Fit the model to the data.


        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe of shape (n_samples, n_features), sensitive attribute must be in the first column
        y : pandas.Series
            Labels array-like of shape (n_samples), must be (0 or 1)

        Returns
        -------
        XtremeFair
            Fitted model
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
            obj=dual_obj(self.fair_weight, self.group_losses),
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

    def score(self, X, y):
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
        return perf * self.alpha + (1 - self.alpha) * fair


def ks_threshold(y_true, y_score):
    """Identify the threshold that maximizes the Kolmogorov-Smirnov statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    return opt_threshold
