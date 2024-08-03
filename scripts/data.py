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
    ],
    "enem": ["racebin", "sexbin"],
    "enem_reg": ["racebin", "sexbin"],
    "enem_large": ["racebin", "sexbin"],
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
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
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
    Y = 1 - df["DEFAULT"]
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


def load_enem():
    df = pd.read_pickle("../data/enem-50000-20.pkl").reset_index(drop=True)
    Y = df["gradebin"].astype(int)
    X = df.drop(columns=["gradebin"])
    for col in X.columns:
        if col in CAT_FEATURES["enem"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_enem_reg():
    df = pd.read_csv("../data/enem_reg_preprocessed.csv").reset_index(drop=True)
    Y = df["gradescore"].astype(float)
    X = df.drop(columns=["gradescore"])
    for col in X.columns:
        if col in CAT_FEATURES["enem"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def load_enem_large():
    df = pd.read_csv("../data/enem_large_preprocessed.csv").reset_index(drop=True)
    Y = df["gradebin"].astype(int)
    X = df.drop(columns=["gradebin"])
    for col in X.columns:
        if col in CAT_FEATURES["enem"]:
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
    elif dataset == "enem":
        return load_enem()
    elif dataset == "enem_reg":
        return load_enem_reg()
    elif dataset == "enem_large":
        return load_enem_large()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def preprocess_dataset(dataset, X_train, X_val, X_test):
    if dataset not in NUM_FEATURES:
        NUM_FEATURES[dataset] = [
            col for col in X_train.columns if col not in CAT_FEATURES[dataset]
        ]

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


def get_subgroup_feature(dataset, X, n_groups=2):
    # assert n_groups in [2, 4, 8]
    if n_groups == 2:
        if dataset == "german":
            A = X.Gender.astype(str)
        elif dataset == "adult":
            A = X.sex.astype(str)
        elif dataset == "compas":
            A = X.race == "Caucasian"
        elif dataset == "acsincome":
            A = X.SEX.astype(str)
        elif dataset == "taiwan":
            A = X.SEX.astype(str)
        elif dataset == "enem" or dataset == "enem_reg":
            A = X.racebin.astype(str)

    elif n_groups == 4:
        if dataset == "german":
            A = X.Gender.astype(str) + "_" + (X.Age > 50).astype(str)
        elif dataset == "compas":
            A = (
                (X.race == "Caucasian").astype(str)
                + "_"
                + ((X.age_cat == "25 - 45") | (X.age_cat == "Less than 25")).astype(str)
            )
        elif dataset == "adult":
            A = X.sex.astype(str) + "_" + (X.age > 50).astype(str)
        elif dataset == "taiwan":
            A = X.SEX.astype(str) + "_" + (X.AGE > 50).astype(str)
        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "1"
                elif race == "african_america":
                    return "2"
                elif race == "asian":
                    return "3"
                else:
                    return "4"

            A = X.RAC1P.apply(race_cat)
        elif dataset == "enem" or dataset == "enem_reg":
            A = X.racebin.astype(str) + "_" + X.sexbin.astype(str)

    elif n_groups == 8:
        if dataset == "german":

            def age_cat(age):
                if age < 30:
                    return "1"
                elif age < 40:
                    return "2"
                elif age < 50:
                    return "3"
                else:
                    return "4"

            A = X.Gender.astype(str) + "_" + X.Age.apply(age_cat).astype(str)
        elif dataset == "adult":

            def age_cat(age):
                if age < 30:
                    return "1"
                elif age < 40:
                    return "2"
                elif age < 50:
                    return "3"
                else:
                    return "4"

            A = X.sex.astype(str) + "_" + X.age.apply(age_cat).astype(str)

        elif dataset == "taiwan":

            def age_cat(age):
                if age < 30:
                    return "1"
                elif age < 40:
                    return "2"
                elif age < 50:
                    return "3"
                else:
                    return "4"

            A = X.SEX.astype(str) + "_" + X.AGE.apply(age_cat).astype(str)

        elif dataset == "compas":

            def race_cat(race):
                if race == "African-American" or race == "Hispanic":
                    return "1"
                elif race == "Caucasian":
                    return "2"
                elif race == "Asian":
                    return "3"
                else:
                    return "4"

            A = (
                X.race.apply(race_cat)
                + "_"
                + ((X.age_cat == "25 - 45") | (X.age_cat == "Less than 25")).astype(str)
            )

        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "1"
                elif race == "african_america":
                    return "2"
                elif race == "asian":
                    return "3"
                else:
                    return "4"

            A = X.RAC1P.apply(race_cat) + "_" + X.SEX.astype(str)
        elif dataset == "enem" or dataset == "enem_reg":
            # transform age into 2 categories
            age = X[[f"TP_FAIXA_ETARIA_{i}" for i in range(1, 10)]].sum(axis=1)
            A = (
                X.racebin.astype(str)
                + "_"
                + X.sexbin.astype(str)
                + "_"
                + age.astype(str)
            )

    elif n_groups > 20:
        assert dataset == "enem_large"
        sg_columns = [col for col in X.columns if "SG_UF_PROVA" in col]
        # A is the index of the non zero column
        A = X[sg_columns].idxmax(axis=1)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A.unique())])
    print(sensitive_map)
    A = A.map(sensitive_map)
    return A


def get_fold(dataset, fold, n_folds=10, n_groups=2, random_state=None):
    X, Y = load_dataset(dataset)
    A = get_subgroup_feature(dataset, X, n_groups)
    if dataset == "enem_reg":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
        if i == fold:
            X_train, A_train, Y_train = (
                X.iloc[train_index],
                A.iloc[train_index],
                Y.iloc[train_index],
            )
            X_train, X_val, A_train, A_val, Y_train, Y_val = train_test_split(
                X_train,
                A_train,
                Y_train,
                test_size=1 / (n_folds - 1),
                random_state=random_state,
                stratify=A_train,
            )
            X_test, A_test, Y_test = (
                X.iloc[test_index],
                A.iloc[test_index],
                Y.iloc[test_index],
            )
            X_train, X_val, X_test = preprocess_dataset(dataset, X_train, X_val, X_test)

            return (
                X_train,
                A_train,
                Y_train,
                X_val,
                A_val,
                Y_val,
                X_test,
                A_test,
                Y_test,
            )


def get_fold_holdout(dataset, fold, n_folds=10, n_groups=2, random_state=None):
    X, Y = load_dataset(dataset)
    A = get_subgroup_feature(dataset, X, n_groups)
    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
        X, Y, A, test_size=0.2, random_state=random_state
    )
    if dataset == "enem_reg":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_index, val_index) in enumerate(kf.split(X_train, Y_train)):
        if i == fold:
            X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            A_train, A_val = A_train.iloc[train_index], A_train.iloc[val_index]
            Y_train, Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
            X_train, X_val, X_test = preprocess_dataset(dataset, X_train, X_val, X_test)
            return (
                X_train,
                A_train,
                Y_train,
                X_val,
                A_val,
                Y_val,
                X_test,
                A_test,
                Y_test,
            )
