PARAM_SPACES = {
    "M2FGBClassifier": {
        "min_child_weight": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "n_estimators": {"type": "int", "low": 20, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 1e-2, "high": 0.5, "log": True},
        "num_leaves": {"type": "int", "low": 2, "high": 64},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
        "fair_weight": {"type": "float", "low": 1e-3, "high": 1, "log" : True},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 1e-3,
            "high": 0.5,
            "log": True,
        },
    },
    "FairGBMClassifier": {
        "min_child_weight": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "n_estimators": {"type": "int", "low": 20, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 1e-2, "high": 0.5, "log": True},
        "num_leaves": {"type": "int", "low": 2, "high": 64},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
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
        "min_child_weight": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        "n_estimators": {"type": "int", "low": 20, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 1e-2, "high": 0.5, "log": True},
        "num_leaves": {"type": "int", "low": 2, "high": 64},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 1000, "log": True},
    },
    "MinMaxFair": {
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "gamma": {"type": "float", "low": 0, "high": 1},
        "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
        "a": {"type": "float", "low": 0.1, "high": 1},
        "b": {"type": "float", "low": 1e-2, "high": 1},
    },
    "MinMaxFairRegressor": {
        "n_estimators": {"type": "int", "low": 10, "high": 500, "log": True},
        "gamma": {"type": "float", "low": 0, "high": 1},
        "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
        "a": {"type": "float", "low": 0.1, "high": 1},
        "b": {"type": "float", "low": 1e-2, "high": 1},
    },
    "MinimaxPareto": {
        "n_iterations": {"type": "int", "low": 10, "high": 500, "log": True},
        "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
        "alpha": {"type": "float", "low": 0.1, "high": 0.9},
        "Kmin": {"type": "int", "low": 10, "high": 50},
    },
    "LGBMRegressor": {
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "min_child_weight": {"type": "float", "low": 1, "high": 1e4, "log": True},
        "n_estimators": {"type": "int", "low": 20, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
    },
    "M2FGBRegressor": {
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "min_child_weight": {"type": "float", "low": 1, "high": 1e4, "log": True},
        "n_estimators": {"type": "int", "low": 20, "high": 500, "log": True},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.5, "log": True},
        "multiplier_learning_rate": {
            "type": "float",
            "low": 1e-3,
            "high": 0.5,
            "log": True,
        },
        "fair_weight": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    },
}

PARAM_SPACES_ACSINCOME = PARAM_SPACES.copy()
PARAM_SPACES_ACSINCOME["MinMaxFair"] = {
    "n_estimators": {"type": "int", "low": 10, "high": 50, "log": True},
    "gamma": {"type": "float", "low": 0, "high": 1},
    "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
    "a": {"type": "float", "low": 0.1, "high": 1},
    "b": {"type": "float", "low": 1e-2, "high": 1},
    "max_iter": {"type": "int", "low": 10, "high": 11},
}
PARAM_SPACES_ACSINCOME["MinimaxPareto"] = {
    "n_iterations": {"type": "int", "low": 10, "high": 50, "log": True},
    "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
    "alpha": {"type": "float", "low": 0.1, "high": 0.9},
    "Kmin": {"type": "int", "low": 10, "high": 50},
    "max_iter": {"type": "int", "low": 10, "high": 11},
}
PARAM_SPACES_ACSINCOME["MinMaxFairRegressor"] = {
    "n_estimators": {"type": "int", "low": 10, "high": 50, "log": True},
    "gamma": {"type": "float", "low": 0, "high": 1},
    "C": {"type": "float", "low": 1e-4, "high": 1e4, "log": True},
    "a": {"type": "float", "low": 0.1, "high": 1},
    "b": {"type": "float", "low": 1e-2, "high": 1},
    "max_iter": {"type": "int", "low": 10, "high": 11},
}
