from typing import List, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import log_loss

from m2fgb import utils
import lightgbm as lgb


class M2FGB(BaseEstimator):
    def __init__(
        self,
        fairness_constraint: str = "equalized_loss",
        objective: str = "binary",
        fair_weight: float = 0.5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        multiplier_learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state=None,
    ) -> None:
        assert fairness_constraint in [
            "equalized_loss",
            "true_positive_rate",
            "positive_rate",
            "true_negative_rate",
        ]

        assert objective in ["binary", "mse"]

        assert fair_weight >= 0 and fair_weight <= 1

        self.fairness_constraint = fairness_constraint
        self.objective = objective
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.multiplier_learning_rate = multiplier_learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_attribute: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None,
        sensitive_attribute_val: Union[np.ndarray, pd.Series] = None,
    ) -> "M2FGB":
        """Fit the model to the data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Dataframe of shape (n_samples, n_features)
        y : pandas.Series or numpy.ndarray
            Labels array-like of shape (n_samples), must be (0 or 1)
        sensitive_attribute : pandas.Series or numpy.ndarray
            Sensitive attribute array-like of shape (n_samples)
        X_val : pandas.DataFrame or numpy.ndarray, optional
            Validation dataframe of shape (n_samples, n_features)
        y_val : pandas.Series or numpy.ndarray, optional
            Validation labels array-like of shape (n_samples), must be (0 or 1)
        sensitive_attribute_val : pandas.Series or numpy.ndarray, optional
            Validation sensitive attribute array-like of shape (n_samples)

        Returns
        -------
        M2FGB
            Fitted model
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if isinstance(sensitive_attribute, pd.Series):
            sensitive_attribute = sensitive_attribute.to_numpy()
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()
        if sensitive_attribute_val is not None and isinstance(
            sensitive_attribute_val, pd.Series
        ):
            sensitive_attribute_val = sensitive_attribute_val.to_numpy()

        if self.objective == "mse":
            self.dual_obj = dual_obj_reg
            self.eval_metric = reg_eval_metric
        else:
            self.dual_obj = dual_obj_cls
            self.eval_metric = cls_eval_metric

        # sort based in sensitive_attribute
        idx = np.argsort(sensitive_attribute)
        X = X[idx]
        y = y[idx]
        sensitive_attribute = sensitive_attribute[idx]

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.info = []
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": self.dual_obj(
                sensitive_attribute,
                self.fair_weight,
                self.info,
                self.fairness_constraint,
                self.multiplier_learning_rate,
            ),
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbose": -1,
        }
        if self.random_state is not None:
            params["random_seed"] = self.random_state

        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val)
            self.model_ = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=self.n_estimators,
                feval=self.eval_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, min_delta=1e-5),
                ],
            )
        else:
            self.model_ = lgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
            )
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Base method to predict the labels of the data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Dataframe of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,)
        """
        raise NotImplementedError("This method should be implemented in the subclasses.")


class M2FGBClassifier(M2FGB, ClassifierMixin):
    def __init__(
        self,
        fairness_constraint: str = "equalized_loss",
        fair_weight: float = 0.5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        multiplier_learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state=None,
    ):
        super().__init__(
            fairness_constraint=fairness_constraint,
            objective="binary",
            fair_weight=fair_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            multiplier_learning_rate=multiplier_learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.dual_obj = dual_obj_cls
        self.eval_metric = cls_eval_metric

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict the labels of the data."""
        check_is_fitted(self)
        X = check_array(X)
        log_odds = self.model_.predict(X)
        log_odds = np.asarray(log_odds)  # Ensure log_odds is a NumPy array
        preds = 1 / (1 + np.exp(-1 * log_odds))
        return (preds > 0.5).astype(int)

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        start_iteration: int = 0,
        num_iteration: Union[Any, int] = None,
    ):
        """Predict the probabilities of the data."""
        check_is_fitted(self)
        X = check_array(X)
        log_odds = self.model_.predict(
            X, start_iteration=start_iteration, num_iteration=num_iteration
        )
        log_odds = np.asarray(log_odds)  # Ensure log_odds is a NumPy array
        preds_pos = 1 / (1 + np.exp(-log_odds))
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds


class M2FGBRegressor(M2FGB, RegressorMixin):
    def __init__(
        self,
        fairness_constraint: str = "equalized_loss",
        fair_weight: float = 0.5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        multiplier_learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state=None,
    ):
        super().__init__(
            fairness_constraint=fairness_constraint,
            objective="mse",
            fair_weight=fair_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            multiplier_learning_rate=multiplier_learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict with the data."""
        check_is_fitted(self)
        X = check_array(X)
        preds = self.model_.predict(X)
        preds = np.asarray(preds)
        return preds


def cls_eval_metric(preds, dtrain) -> Tuple[str, float, bool]:
    """Custom evaluation metric for classification tasks."""
    preds = 1 / (1 + np.exp(-preds))
    labels = dtrain.get_label()
    l1 = log_loss(labels, preds)
    l2 = utils.max_logloss_score(labels, preds, dtrain.get_weight())
    l = (1 - dtrain.get_weight()) * l1 + dtrain.get_weight() * l2
    return "logloss", l, False


def dual_obj_cls(
    subgroup: np.ndarray,
    fair_weight: float,
    info: List[Dict[str, Any]],
    fairness_constraint: str = "equalized_loss",
    multiplier_learning_rate: float = 0.1,
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
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, used only if dual_learning="gradient", by default 0.1
    """
    I = utils.get_subgroup_indicator(subgroup)
    mu = np.ones(I.shape[1])
    mu = mu / np.sum(mu) * fair_weight
    info.append(
        {
            "loss": np.ones(I.shape[1]),
            "mu": mu,
            "epsilon": 1,
        }
    )

    def custom_obj(predt, dtrain):
        y_true = dtrain.get_label()
        y_pred = 1 / (1 + np.exp(-predt))
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # avoid log(0)
        loss_group = utils.logloss_group(y_pred, y_true, subgroup, fairness_constraint)
        epsilon = np.max(loss_group)
        mu = info[-1]["mu"].copy()

        mu_new = mu + multiplier_learning_rate * fair_weight * (loss_group - epsilon)
        mu_new = utils.projection_to_simplex(mu_new, z=fair_weight)

        info.append(
            {
                "loss": loss_group,
                "mu": mu_new,
                "epsilon": epsilon,
            }
        )
        grad_fair = utils.logloss_group_grad(y_pred, y_true, fairness_constraint)
        grad_fair = I.multiply(grad_fair.reshape(-1, 1)) @ mu

        hess_fair = utils.logloss_group_hess(y_pred, y_true, fairness_constraint)
        hess_fair = I.multiply(hess_fair.reshape(-1, 1)) @ mu

        grad = utils.logloss_grad(y_pred, y_true)
        hess = utils.logloss_hessian(y_pred, y_true)

        grad = (1 - fair_weight) * grad + grad_fair
        hess = (1 - fair_weight) * hess + hess_fair
        return grad, hess

    return custom_obj


def reg_eval_metric(preds, dtrain) -> Tuple[str, float, bool]:
    """Custom evaluation metric for regression tasks."""
    preds = np.asarray(preds)  # Ensure preds is a NumPy array
    y_true = dtrain.get_label()
    l1 = np.mean((y_true - preds) ** 2)
    l2 = utils.max_mse(y_true, preds, dtrain.get_weight())
    l = (1 - dtrain.get_weight()) * l1 + dtrain.get_weight() * l2
    return "logloss", l, False


def dual_obj_reg(
    subgroup: np.ndarray,
    fair_weight: float,
    info: List[Dict[str, Any]],
    fairness_constraint: str = "",
    multiplier_learning_rate: float = 0.1,
):
    """This helper function will define a custom objective function using the fair_weight parameter.

    Parameters
    ----------
    soubgroup : ndarray
        Array with the subgroup labels.
    fair_weight : float
        Weight of the fairness term in the loss function.
    info: list
        List where the losses for each subgroup will be stored.
    fairness_constraint: str, optional
        Fairness constraint used in learning. It is not used in regression, but kept for consistency.
    multiplier_learning_rate: float, optional
        Learning rate used in the gradient learning of the dual, by default 0.1
    """
    I = utils.get_subgroup_indicator(subgroup)
    mu = np.ones(I.shape[1])
    mu = mu / np.sum(mu) * fair_weight
    info.append(
        {
            "loss": np.ones(I.shape[1]),
            "mu": mu,
            "epsilon": 1,
        }
    )

    def custom_obj(predt, dtrain):
        y_true = dtrain.get_label()
        y_pred = predt
        loss_group = utils.squaredloss_group(y_pred, y_true, subgroup)
        epsilon = np.max(loss_group)
        mu = info[-1]["mu"].copy()

        mu_new = mu + multiplier_learning_rate * (loss_group - epsilon)
        mu_new = utils.projection_to_simplex(mu_new, z=fair_weight)

        info.append({"loss": loss_group, "mu": mu_new, "epsilon": epsilon})
        grad_fair = utils.squaredloss_grad(y_pred, y_true)
        grad_fair = I.multiply(grad_fair.reshape(-1, 1)) @ mu

        hess_fair = utils.squaredloss_hess(y_pred, y_true)
        hess_fair = I.multiply(hess_fair.reshape(-1, 1)) @ mu

        grad = utils.squaredloss_grad(y_pred, y_true)
        hess = utils.squaredloss_hess(y_pred, y_true)
        grad = (1 - fair_weight) * grad + grad_fair
        hess = (1 - fair_weight) * hess + hess_fair

        return grad, hess

    return custom_obj
