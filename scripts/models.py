import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.sparse import csr_matrix

import utils
import lightgbm as lgb
import fairgbm

import sys

sys.path.append("../minimax-fair")
import src.minmaxML as mmml

sys.path.append("../MMPF")
from MinimaxParetoFair.MMPF_trainer import APSTAR, SKLearn_Weighted_LLR

lgb.register_logger(utils.CustomLogger())
fairgbm.register_logger(utils.CustomLogger())


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


def logloss_grad(y_pred, y_true):
    """Compute the gradient for cross entropy log loss."""
    grad = -(y_true - y_pred)
    return grad


def logloss_hessian(y_pred, y_true):
    """Compute the hessian for cross entropy log loss."""
    hess = y_pred * (1 - y_pred)
    return hess


def logloss_group(y_pred, y_true, subgroup, fairness_constraint):
    """For each subgroup, calculates the mean log loss of the samples."""
    if fairness_constraint == "equalized_loss":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if fairness_constraint == "positive_rate":
        y_ = np.ones(y_true.shape[0])  # all positive class
        loss = -(y_ * np.log(y_pred) + (1 - y_) * np.log(1 - y_pred))
    elif fairness_constraint == "true_positive_rate":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss[y_true == 0] = np.nan  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss[y_true == 1] = np.nan  # only consider the loss of the positive class

    # TODO LATER: use I (indicator) to calculate averages

    # smart numpy groupby that assumes that subgroup is sorted
    loss = np.column_stack((loss, subgroup))
    loss = np.split(loss[:, 0], np.unique(loss[:, 1], return_index=True)[1][1:])
    loss = np.array([np.nanmean(l) for l in loss])
    return loss


def logloss_group_grad(y_pred, y_true, fairness_constraint):
    """Create an array with the gradient of fairness metrics."""
    if fairness_constraint == "equalized_loss":
        grad = -(y_true - y_pred)
    elif fairness_constraint == "positive_rate":
        y_ = np.ones(y_true.shape[0])  # all positive class
        grad = -(y_ - y_pred)
    elif fairness_constraint == "true_positive_rate":
        grad = -(y_true - y_pred)
        grad[y_true == 0] = 0  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        grad = -(y_true - y_pred)
        grad[y_true == 1] = 0  # only consider the loss of the negative class

    return grad


def logloss_group_hess(y_pred, y_true, fairness_constraint):
    """Create an array with the hessian of fairness metrics."""
    if (
        fairness_constraint == "equalized_loss"
        or fairness_constraint == "positive_rate"
    ):
        hess = y_pred * (1 - y_pred)
    elif fairness_constraint == "true_positive_rate":
        hess = y_pred * (1 - y_pred)
        hess[y_true == 0] = 0  # only consider the loss of the positive class
    elif fairness_constraint == "true_negative_rate":
        hess = y_pred * (1 - y_pred)
        hess[y_true == 1] = 0

    return hess


def get_subgroup_indicator(subgroup):
    groups = np.unique(subgroup)
    n = len(subgroup)
    I = np.zeros((subgroup.shape[0], len(groups)))
    n_g_max = -np.inf
    for i, g in enumerate(groups):
        n_g = np.sum(subgroup == g)
        n_g_max = max(n_g_max, n_g)
        I[subgroup == g, i] = 1 / np.sum(subgroup == g)

    I = I * n
    I = csr_matrix(I)
    return I


def dual_obj_cls(
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
    I = get_subgroup_indicator(subgroup)

    def custom_obj(predt, dtrain):
        y_true = dtrain.get_label()
        y_pred = 1 / (1 + np.exp(-predt))
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # avoid log(0)
        loss_group = logloss_group(y_pred, y_true, subgroup, fairness_constraint)
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
                mu_opt = loss_group
            else:
                mu_opt = mu_opt_list[-1].copy()

            mu_opt += multiplier_learning_rate * loss_group
            mu_opt = projection_to_simplex(mu_opt, z=fair_weight)

        elif dual_learning == "gradient_norm2":
            if mu_opt_list[0] is None:
                mu_opt = loss_group
            else:
                mu_opt = mu_opt_list[-1].copy()

            mu_opt += multiplier_learning_rate * loss_group
            mu_opt = mu_opt / np.sum(mu_opt) * fair_weight

        if mu_opt_list[0] is None:
            mu_opt_list[0] = mu_opt
        else:
            mu_opt_list.append(mu_opt)

        grad_fair = logloss_group_grad(y_pred, y_true, fairness_constraint)
        # grad_fair = I * grad_fair.reshape(-1, 1) @ mu_opt
        grad_fair = I.multiply(grad_fair.reshape(-1, 1)) @ mu_opt

        hess_fair = logloss_group_hess(y_pred, y_true, fairness_constraint)
        hess_fair = I.multiply(hess_fair.reshape(-1, 1)) @ mu_opt
        # hess_fair = I * hess_fair.reshape(-1, 1) @ mu_opt

        grad = logloss_grad(y_pred, y_true)
        hess = logloss_hessian(y_pred, y_true)

        # It is not necessary to multiply fairness gradient by fair_weight because it is already included on mu
        # grad = (1 - fair_weight) * grad + fair_weight * grad_fair
        # hess = (1 - fair_weight) * hess + fair_weight * hess_fair

        grad = (1 - fair_weight) * grad + grad_fair
        hess = (1 - fair_weight) * hess + hess_fair

        return grad, hess

    return custom_obj


class M2FGBClassifier(BaseEstimator, ClassifierMixin):
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
        dual_learning="gradient_norm2",
        multiplier_learning_rate=0.1,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
            "true_positive_rate",
            "positive_rate",
            "true_negative_rate",
        ]
        assert dual_learning in ["optim", "gradient", "gradient_norm", "gradient_norm2"]

        assert fair_weight >= 0 and fair_weight <= 1

        self.fairness_constraint = fairness_constraint
        self.dual_learning = dual_learning
        self.multiplier_learning_rate = multiplier_learning_rate
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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
        X,
        y,
        sensitive_attribute,
        X_val=None,
        Y_val=None,
        sensitive_attribute_val=None,
    ):
        """Fit the model to the data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
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
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(sensitive_attribute, pd.Series):
            sensitive_attribute = sensitive_attribute.values

        # sort based in sensitive_attribute
        idx = np.argsort(sensitive_attribute)
        X = X[idx]
        y = y[idx]
        sensitive_attribute = sensitive_attribute[idx]

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.group_losses = []
        self.mu_opt_list = [None]
        dtrain = lgb.Dataset(X, label=y)
        if X_val is not None:
            sensitive_attribute_val = sensitive_attribute_val.values
            dval = lgb.Dataset(X_val.values, label=Y_val.values)

        def custom_metric(preds, val_data):
            preds = 1 / (1 + np.exp(-preds))
            labels = val_data.get_label()
            l1 = log_loss(labels, preds)
            l2 = utils.max_logloss_score(labels, preds, sensitive_attribute_val)
            return (
                "obj_function",
                (1 - self.fair_weight) * l1 + self.fair_weight * l2,
                False,
            )

        params = {
            "objective": dual_obj_cls(
                sensitive_attribute,
                self.fair_weight,
                self.group_losses,
                self.mu_opt_list,
                self.fairness_constraint,
                self.dual_learning,
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
            self.model_ = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=self.n_estimators,
                feval=custom_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=25),
                ],
            )
        else:
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


class LGBMClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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
        X,
        y,
        sensitive_attribute,
        X_val=None,
        Y_val=None,
        sensitive_attribute_val=None,
    ):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(sensitive_attribute, pd.Series):
            sensitive_attribute = sensitive_attribute.values

        # sort based in sensitive_attribute
        idx = np.argsort(sensitive_attribute)
        X = X[idx]
        y = y[idx]
        sensitive_attribute = sensitive_attribute[idx]

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": "binary",
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

        def custom_metric(preds, val_data):
            preds = 1 / (1 + np.exp(-preds))
            labels = val_data.get_label()
            l1 = log_loss(labels, preds)
            return "obj_function", l1, False

        if X_val is not None:
            dval = lgb.Dataset(X_val.values, label=Y_val.values)
            self.model_ = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=self.n_estimators,
                feval=custom_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=25),
                ],
            )

        else:
            self.model_ = lgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
            )
        return self

    def predict(self, X):
        """Predict the labels of the data."""
        check_is_fitted(self)
        X = check_array(X)
        preds = self.model_.predict(X)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict the probabilities of the data."""
        check_is_fitted(self)
        X = check_array(X)
        preds_pos = self.model_.predict(X)
        preds = np.ones((preds_pos.shape[0], 2))
        preds[:, 1] = preds_pos
        preds[:, 0] -= preds_pos
        return preds


class FairGBMClassifier(fairgbm.FairGBMClassifier):
    def fit(self, X, Y, A):
        super().fit(X, Y, constraint_group=A)
        return self


class MinMaxFair(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        fairness_constraint="equalized_loss",
        a=1,
        b=0.5,
        gamma=0.0,
        relaxed=False,
        penalty="l2",
        C=1.0,
        max_iter=100,
    ):
        assert fairness_constraint in ["equalized_loss", "tpr"]
        self.fairness_constraint = fairness_constraint
        self.n_estimators = n_estimators
        self.a = a
        self.b = b
        self.gamma = gamma
        self.relaxed = relaxed
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        error_type = "Log-Loss"
        if self.fairness_constraint == "tpr":
            error_type = "FN-Log-Loss"

        # train a logistic model to get min and max logloss
        model = LogisticRegression(
            penalty=self.penalty,
            C=1 if self.penalty == "none" else self.C,
            max_iter=self.max_iter,
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
            error_type=error_type,
            extra_error_types=set(),
            pop_error_type=error_type,
            # data transform
            rescale_features=False,
            test_size=0.0,
            random_split_seed=0,
            # model selection
            model_type="LogisticRegression",
            # parameters related to LR model
            max_logi_iters=self.max_iter,
            tol=1e-8,
            fit_intercept=True,
            logistic_solver="saga",
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

        predictions = np.stack([model.predict_proba(X)[:, 1] for model in self.model])
        idx = np.random.choice(predictions.shape[0], predictions.shape[1], replace=True).reshape(1, -1)
        predictions = np.take_along_axis(predictions, idx, axis=0)
        predictions = np.squeeze(predictions, axis=0)  # remove the extra dimension
        predictions = np.stack([1-predictions, predictions], axis=1)
        
        # def rand_pred(row):
        #     idx = np.random.choice(len(self.model))
        #     return self.model[idx].predict_proba(row.reshape(1, -1))

        # predictions = np.apply_along_axis(rand_pred, 1, X)
        # predictions = np.squeeze(predictions, axis=1)  # remove the extra dimension
        return predictions


class MinimaxPareto(BaseEstimator, ClassifierMixin):
    def __init__(
        self, n_iterations=100, C=1.0, Kini=1, Kmin=20, alpha=0.5, max_iter=100, fairness_constraint = "equalized_loss"
    ):
        self.n_iterations = n_iterations
        self.C = C
        self.Kini = Kini
        self.Kmin = Kmin
        self.alpha = alpha
        self.max_iter = max_iter
        self.fairness_constraint = fairness_constraint

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        # insert sensitive attribute as first column of the dataframe
        self.classes_ = np.unique(y)

        model = SKLearn_Weighted_LLR(
            X,
            y,
            sensitive_attribute,
            X,
            y,
            sensitive_attribute,
            C_reg=self.C,
            max_iter=self.max_iter,
            fairness_constraint=self.fairness_constraint
        )

        mu_ini = np.ones(len(sensitive_attribute.unique()))
        mu_ini /= mu_ini.sum()

        results = APSTAR(
            model,
            mu_ini,
            niter=self.n_iterations,
            max_patience=self.n_iterations // 2,
            Kini=1,
            Kmin=self.Kmin,
            alpha=self.alpha,
            verbose=False,
        )

        mu_best = results["mu_best_list"][-1]
        model.weighted_fit(X, y, sensitive_attribute, mu_best)
        self.model = model
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        predictions = self.predict_proba(X)[:, 1]
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model.predict_proba(X)


class M2FGBRegressor(BaseEstimator, RegressorMixin):
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
        dual_learning="gradient_norm",
        multiplier_learning_rate=0.1,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
    ):
        assert fairness_constraint in [
            "equalized_loss",
        ]
        assert dual_learning in ["optim", "gradient", "gradient_norm", "gradient_norm2"]

        assert fair_weight >= 0 and fair_weight <= 1

        self.fairness_constraint = fairness_constraint
        self.dual_learning = dual_learning
        self.multiplier_learning_rate = multiplier_learning_rate
        self.fair_weight = fair_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y, sensitive_attribute):
        """Fit the model to the data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
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
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(sensitive_attribute, pd.Series):
            sensitive_attribute = sensitive_attribute.values

        n_g = len(np.unique(sensitive_attribute))
        min_child_weight = self.min_child_weight
        # min_child_weight *= (1 - self.fair_weight) + self.fair_weight/n_g # trick to scale min_child_weight with hessian

        # sort based in sensitive_attribute
        idx = np.argsort(sensitive_attribute)
        X = X[idx]
        y = y[idx]
        sensitive_attribute = sensitive_attribute[idx]

        X, y = check_X_y(X, y)
        self.range_ = [np.min(y), np.max(y)]
        self.group_losses = []
        self.mu_opt_list = [None]
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": dual_obj_reg(
                sensitive_attribute,
                self.fair_weight,
                self.group_losses,
                self.mu_opt_list,
                self.fairness_constraint,
                self.dual_learning,
                self.multiplier_learning_rate,
            ),
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": min_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
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
        preds = self.model_.predict(X)
        return preds


def dual_obj_reg(
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
    I = get_subgroup_indicator(subgroup)

    def custom_obj(predt, dtrain):
        y_true = dtrain.get_label()
        y_pred = predt
        loss_group = squaredloss_group(y_pred, y_true, subgroup)
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
                mu_opt = loss_group
            else:
                mu_opt = mu_opt_list[-1].copy()

            mu_opt += multiplier_learning_rate * loss_group
            mu_opt = projection_to_simplex(mu_opt, z=fair_weight)

        elif dual_learning == "gradient_norm2":
            if mu_opt_list[0] is None:
                mu_opt = np.ones(loss_group.shape[0])
            else:
                mu_opt = mu_opt_list[-1].copy()

            mu_opt += multiplier_learning_rate * loss_group
            mu_opt = mu_opt / np.sum(mu_opt) * fair_weight

        if mu_opt_list[0] is None:
            mu_opt_list[0] = mu_opt
        else:
            mu_opt_list.append(mu_opt)

        grad_fair = squaredloss_grad(y_pred, y_true, fairness_constraint)
        grad_fair = I.multiply(grad_fair.reshape(-1, 1)) @ mu_opt

        hess_fair = squaredloss_hess(y_pred, y_true, fairness_constraint)
        hess_fair = I.multiply(hess_fair.reshape(-1, 1)) @ mu_opt

        grad = squaredloss_grad(y_pred, y_true)
        hess = squaredloss_hess(y_pred, y_true)
        grad = (1 - fair_weight) * grad + grad_fair
        hess = (1 - fair_weight) * hess + hess_fair

        return grad, hess

    return custom_obj


def squaredloss_group(y_pred, y_true, subgroup, fairness_constraint="equalized_loss"):
    """For each subgroup, calculates the mean log loss of the samples."""
    loss = (y_true - y_pred) ** 2

    # TODO LATER: use I (indicator) to calculate averages

    # smart numpy groupby that assumes that subgroup is sorted
    loss = np.column_stack((loss, subgroup))
    loss = np.split(loss[:, 0], np.unique(loss[:, 1], return_index=True)[1][1:])
    loss = np.array([np.mean(l) for l in loss])
    return loss


def squaredloss_grad(y_pred, y_true, fairness_constraint=""):
    grad = 2 * (y_pred - y_true)
    return grad


def squaredloss_hess(y_pred, y_true, fairness_constraint=""):
    hess = 2 * np.ones(y_pred.shape[0])
    return hess


class LGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_gain_to_split=0.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        self.random_state = random_state

    def fit(self, X, y, sensitive_attribute):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(sensitive_attribute, pd.Series):
            sensitive_attribute = sensitive_attribute.values

        # sort based in sensitive_attribute
        idx = np.argsort(sensitive_attribute)
        X = X[idx]
        y = y[idx]
        sensitive_attribute = sensitive_attribute[idx]

        X, y = check_X_y(X, y)
        self.range_ = [np.min(y), np.max(y)]
        dtrain = lgb.Dataset(X, label=y)

        params = {
            "objective": "regression",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_gain_to_split": self.min_gain_to_split,
            "verbose": -1,
        }
        if self.random_state is not None:
            params["random_seed"] = self.random_state

        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
        return self

    def predict(self, X):
        """Predict the labels of the data."""
        check_is_fitted(self)
        X = check_array(X)
        preds = self.model_.predict(X)
        return preds


class MinMaxFairRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        fairness_constraint="equalized_loss",
        a=1,
        b=0.5,
        gamma=0.0,
        relaxed=False,
        C = 1.0,
        max_iter = 100,
    ):
        assert fairness_constraint in ["equalized_loss"]
        self.fairness_constraint = fairness_constraint
        self.n_estimators = n_estimators
        self.a = a
        self.b = b
        self.gamma = gamma
        self.relaxed = relaxed
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y, sensitive_attribute):
        X, y = check_X_y(X, y)
        self.range_ = [np.min(y), np.max(y)]

        error_type = "MSE"

        # train a logistic model to get min and max logloss
        model = Ridge(
            solver="saga",
            max_iter=self.max_iter,
            alpha = self.C

        )
        model.fit(X, y)
        y_pred = model.predict(X)
        min_mse = np.inf
        max_mse = -np.inf
        for g in np.unique(sensitive_attribute):
            idx = sensitive_attribute == g
            mse = np.mean((y[idx] - y_pred[idx]) ** 2)
            min_mse = min(min_mse, mse)
            max_mse = max(mse, mse)

        print(min_mse, max_mse, self.gamma)
        gamma_hat = min_mse + self.gamma * (max_mse - min_mse)

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
            error_type=error_type,
            extra_error_types=set(),
            pop_error_type=error_type,
            # data transform
            rescale_features=False,
            test_size=0.0,
            random_split_seed=0,
            # model selection
            model_type="Ridge",
            # parameters related to LR model
            max_logi_iters=self.max_iter,
            tol=1e-8,
            fit_intercept=True,
            logistic_solver="saga",
            penalty="l2",
            C=self.C,
            # parameters related to MLP
            lr=0.01,
            momentum=0.9,
            weight_decay=0,
            n_epochs=10000,
            hidden_sizes=(2 / 3,),
            # parameters related to output
            display_plots=False,
            verbose=True,
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
        predictions = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            predictions[i] = self.model[np.random.choice(len(self.model))].predict(
                X[i].reshape(1, -1)
            )
        return predictions
