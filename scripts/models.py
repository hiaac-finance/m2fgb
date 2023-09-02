import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import roc_auc_score
import xgb

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators = 10,
        eta = 0.3,
        colsample_bytree = 0.5,
        max_depth = 5,
        min_child_weight = 1,
        l2_weight = 1,
        objective = None,
    ):
        self.eta = eta
        self.n_estimators = n_estimators
        self.colsample_bytree = colsample_bytree
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.l2_weight = l2_weight
        self.objective = objective

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': "binary:logistic",
            "max_depth" : self.max_depth,
            "colsample_bytree" : self.colsample_bytree,
            "min_child_weight" : self.min_child_weight,
            "eta" : self.eta,
            "lambda" : self.l2_weight,
            "tree_method" : "hist",
            "seed" : 0
        }
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round = self.n_estimators,
            obj = None if self.objective is None else self.objective
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        dtest = xgb.DMatrix(X)
        preds = self.model_.predict(dtest)
        return (preds > 0.5).astype(int)  # You can adjust the threshold

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
