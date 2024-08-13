import numpy as np
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
    "enem": ["all"],
    "enem_reg": ["all"],
    "enem_large": ["all"],
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
    "enem": ["none"],
    "enem_reg": ["none"],
    "enem_large": ["none"],
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
    # df = pd.read_pickle("../data/enem-50000-20.pkl").reset_index(drop=True)
    df = pd.read_csv("../data/enem_classif_preprocessed.csv").reset_index(drop=True)
    Y = df["gradebin"].astype(int)
    X = df.drop(columns=["gradebin"])
    return X, Y


def load_enem_reg():
    df = pd.read_csv("../data/enem_reg_preprocessed.csv").reset_index(drop=True)
    Y = df["gradescore"].astype(float)
    X = df.drop(columns=["gradescore"])
    return X, Y


def load_enem_large():
    df = pd.read_csv("../data/enem_large_preprocessed.csv").reset_index(drop=True)
    Y = df["gradebin"].astype(int)
    X = df.drop(columns=["gradebin"])
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
    if CAT_FEATURES[dataset][0] == "all":
        cat_feat = X_train.columns
    elif CAT_FEATURES[dataset][0] == "none":
        cat_feat = []
    else:
        cat_feat = CAT_FEATURES[dataset]

    if NUM_FEATURES[dataset][0] == "all":
        num_feat = X_train.columns
    elif NUM_FEATURES[dataset][0] == "none":
        num_feat = []
    else:
        num_feat = NUM_FEATURES[dataset]



    col_trans = ColumnTransformer(
        [
            ("numeric", StandardScaler(), num_feat),
            (
                "categorical",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                ),
                cat_feat,
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
            def race_cat(race):
                if race == "African-American":
                    return "1"
                elif race == "Caucasian":
                    return "2"
                elif race == "Hispanic":
                    return "3"
                else:
                    return "4"

            A = X.race.apply(race_cat).astype(str)
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
    
    elif n_groups == 6:
        if dataset == "german":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                else:
                    return "Older 40"

            A = X.Gender.astype(str) + ", " + X.Age.apply(age_cat).astype(str)
        elif dataset == "adult":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                else:
                    return "Older 40"

            A = X.sex.astype(str) + ", " + X.age.apply(age_cat).astype(str)

        elif dataset == "taiwan":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                else:
                    return "Older 40"

            A = X.SEX.astype(str) + ", " + X.AGE.apply(age_cat).astype(str)

        elif dataset == "compas":

            def race_cat(race):
                if race in ["African-American", "Caucasian"]:
                    return race
                else:
                    return "Other"

            A = (
                X.race.apply(race_cat)
                + ", "
                + ((X.age_cat == "25 - 45")).apply(
                    lambda x: "Between 25 and 45" if x else "Other"
                )
            )

        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "White"
                elif race == "african_america":
                    return "African-American"
                else:
                    return "Other"

            A = X.SEX.astype(str).str.capitalize() + ", " + X.RAC1P.apply(race_cat)
        elif dataset == "enem" or dataset == "enem_reg":

            def race_cat(race):
                if race in ["White", "Brown"]:
                    return race
                else:
                    return "Other"

            A = (
                X.racebin.apply(race_cat)
                + ", "
                + X.sexbin.astype(str).apply(
                    lambda x: "Male" if x == "1.0" else "Female"
                )
            )

    elif n_groups == 8:
        if dataset == "german":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                elif age < 50:
                    return "Under 50"
                else:
                    return "Older 50"

            A = X.Gender.astype(str) + ", " + X.Age.apply(age_cat).astype(str)
        elif dataset == "adult":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                elif age < 50:
                    return "Under 50"
                else:
                    return "Older 50"

            A = X.sex.astype(str) + ", " + X.age.apply(age_cat).astype(str)

        elif dataset == "taiwan":

            def age_cat(age):
                if age < 30:
                    return "Under 30"
                elif age < 40:
                    return "Under 40"
                elif age < 50:
                    return "Under 50"
                else:
                    return "Older 50"

            A = X.SEX.astype(str) + ", " + X.AGE.apply(age_cat).astype(str)

        elif dataset == "compas":

            def race_cat(race):
                if race in ["African-American", "Caucasian", "Hispanic"]:
                    return race
                else:
                    return "Other"

            A = (
                X.race.apply(race_cat)
                + ", "
                + ((X.age_cat == "25 - 45")).apply(
                    lambda x: "Between 25 and 45" if x else "Other"
                )
            )

        elif dataset == "acsincome":

            def race_cat(race):
                if race == "white":
                    return "White"
                elif race == "african_america":
                    return "African-American"
                elif race == "asian":
                    return "Asian"
                else:
                    return "Other"

            A = X.SEX.astype(str).str.capitalize() + ", " + X.RAC1P.apply(race_cat)
        elif dataset == "enem" or dataset == "enem_reg":

            def race_cat(race):
                if race in ["White", "Black", "Brown"]:
                    return race
                else:
                    return "Other"

            A = (
                X.racebin.apply(race_cat)
                + ", "
                + X.sexbin.astype(str).apply(
                    lambda x: "Male" if x == "1.0" else "Female"
                )
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


def get_strat_split(dataset, n_groups=2, test_size=20, random_state=None):
    X, Y = load_dataset(dataset)
    A = get_subgroup_feature(dataset, X, n_groups)
    is_clf = Y.nunique() == 2
    X_train = []
    X_val = []
    X_test = []
    Y_train = []
    Y_val = []
    Y_test = []
    A_train = []
    A_val = []
    A_test = []

    # Stratified split for each subgroup
    for a in np.unique(A):
        X_a = X[A == a]
        Y_a = Y[A == a]
        A_a = A[A == a]
        test_size_ = int(len(X_a) * test_size / 100)
        X_train_a, X_test_a, Y_train_a, Y_test_a, A_train_a, A_test_a = (
            train_test_split(
                X_a,
                Y_a,
                A_a,
                test_size=test_size_,
                random_state=random_state,
                stratify=Y_a if is_clf else None,
            )
        )
        X_train_a, X_val_a, Y_train_a, Y_val_a, A_train_a, A_val_a = train_test_split(
            X_train_a,
            Y_train_a,
            A_train_a,
            test_size=test_size_,
            random_state=random_state,
            stratify=Y_train_a if is_clf else None,
        )
        X_train.append(X_train_a)
        X_val.append(X_val_a)
        X_test.append(X_test_a)
        Y_train.append(Y_train_a)
        Y_val.append(Y_val_a)
        Y_test.append(Y_test_a)
        A_train.append(A_train_a)
        A_val.append(A_val_a)
        A_test.append(A_test_a)

    X_train = pd.concat(X_train)
    X_val = pd.concat(X_val)
    X_test = pd.concat(X_test)
    Y_train = pd.concat(Y_train)
    Y_val = pd.concat(Y_val)
    Y_test = pd.concat(Y_test)
    A_train = pd.concat(A_train)
    A_val = pd.concat(A_val)
    A_test = pd.concat(A_test)

    X_train, X_val, X_test = preprocess_dataset(dataset, X_train, X_val, X_test)

    # shuffle
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train.iloc[idx]
    Y_train = Y_train.iloc[idx]
    A_train = A_train.iloc[idx]
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
