import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CAT_FEATURES = {
    "german": [
        "CheckingAccount",
        "CreditHistory",
        "Purpose",
        "SavingsAccount",
        "EmploymentSince",
        "Gender",
        "OtherDebtors",
        "Property",
        "OtherInstallmentPlans",
        "Housing",
        "Job",
        "Telephone",
        "ForeignWorker",
    ],
    "adult": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    "compas": [
        "sex",
        "age_cat",
        "race",
    ],
    "acsincome": ["COW", "SCHL", "MAR", "RELP", "RAC1P", "SEX"],
    "taiwan": [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ],
}


NUM_FEATURES = {
    "german": [
        "Age",
        "CreditAmount",
        "Dependents",
        "Duration",
        "ExistingCredits",
        "InstallmentRate",
        "ResidenceSince",
    ],
    "adult": ["age", "capital-gain", "capital-loss", "education-num", "hours-per-week"],
    "compas": [
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
    ],
    "acsincome": ["AGEP", "WKHP"],
    "taiwan": [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ],
}


def load_german():
    df = pd.read_csv("../data/german_preprocessed.csv")
    X = df.drop(["GOOD_RISK"], axis=1)
    Y = df["GOOD_RISK"]
    for col in X.columns:
        if col in CAT_FEATURES["german"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_adult():
    df = pd.read_csv("../data/adult_preprocessed.csv")
    X = df.drop(["INCOME"], axis=1)
    Y = df["INCOME"]
    Y = Y.map({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})
    for col in X.columns:
        if col in CAT_FEATURES["adult"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_taiwan():
    df = pd.read_csv("../data/taiwan_preprocessed.csv")
    X = df.drop(["DEFAULT"], axis=1)
    Y = df["DEFAULT"]
    for col in X.columns:
        if col in CAT_FEATURES["taiwan"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_compas():
    df = pd.read_csv("../data/compas_preprocessed.csv")
    X = df.drop(["two_year_recid"], axis=1)
    Y = 1 - df["two_year_recid"]
    for col in X.columns:
        if col in CAT_FEATURES["compas"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_acsincome():
    df = pd.read_csv("../data/acsincome_preprocessed.csv")
    Y = df["PINCP"]
    X = df.drop(columns=["PINCP"])
    for col in X.columns:
        if col in CAT_FEATURES["acsincome"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_dataset(dataset):
    if dataset == "german":
        return load_german()
    elif dataset == "adult":
        return load_adult()
    elif dataset == "compas":
        return load_compas()
    elif dataset == "taiwan":
        return load_taiwan()
    elif dataset == "acsincome":
        return load_acsincome()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def preprocess_dataset(dataset, X_train, X_val, X_test):
    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUM_FEATURES[dataset]),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                CAT_FEATURES[dataset],
            ),
        ],
        verbose_feature_names_out=False,
    )
    col_trans.set_output(transform="pandas")
    preprocess = Pipeline([("preprocess", col_trans)])
    preprocess.fit(X_train)
    X_train = preprocess.transform(X_train)
    X_val = preprocess.transform(X_val)
    X_test = preprocess.transform(X_test)
    return X_train, X_val, X_test


def get_fold(dataset, fold, n_folds=10, random_state=None):
    X, Y = load_dataset(dataset)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
        if i == fold:
            X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=1 / (n_folds - 1), random_state=random_state
            )
            X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]
            return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_fold_holdout(dataset, fold, random_state=None):
    X, Y = load_dataset(dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state
    )

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for i, (train_index, val_index) in enumerate(kf.split(X_train, Y_train)):
        if i == fold:
            X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            Y_train, Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

            return X_train, Y_train, X_val, Y_val, X_test, Y_test
