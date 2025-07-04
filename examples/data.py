from typing import Tuple
import pandas as pd

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_ACSIncome():
    from folktables import ACSDataSource, ACSIncome

    # Dir path
    data_dir = "data/ACSIncome/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    state_list = [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]

    data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=str(data_dir)
    )
    data = data_source.get_data(states=state_list, download=True)
    dataset_details = deepcopy(ACSIncome)
    dataset_details.features.append("ST")

    features, labels, _ = dataset_details.df_to_numpy(data)
    df = pd.DataFrame(data=features, columns=dataset_details.features)
    df[dataset_details.target] = labels

    # reorder columns
    sensitive_col = "SEX"
    state_col = "ST"
    cols_order = [dataset_details.target, sensitive_col] + list(
        set(dataset_details.features) - {sensitive_col, state_col}
    )
    df = df[cols_order]

    mapping = {
        1: "white",
        2: "african_america",
        3: "american_indian",
        4: "alaska_native",
        5: "american_indian_or_alaska_native",
        6: "asian",
        7: "native_hawaiian",
        8: "other_race",
        9: "two_or_more",
    }
    df["RAC1P"] = df["RAC1P"].apply(lambda x: mapping[x])

    mapping = {1: "male", 2: "female"}
    df["SEX"] = df["SEX"].apply(lambda x: mapping[x])

    df["PINCP"] = df["PINCP"].apply(
        lambda x: 1 if x is True else 0 if x is False else x
    )

    # drop columns with many cateogies
    df = df.drop(columns=["OCCP", "POBP"])

    categorical_columns = ["COW", "SCHL", "MAR", "RELP", "RAC1P", "SEX"]
    for col in df.columns:
        if col in categorical_columns:
            df[col] = pd.Categorical(df[col])

    df.to_csv("data/acsincome_preprocessed.csv", index=False)


CAT_FEATURES = {
    "acsincome": ["COW", "SCHL", "MAR", "RELP", "RAC1P", "SEX"],
}


NUM_FEATURES = {
    "acsincome": ["AGEP", "WKHP"],
}


def load_dataset(name : str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../data/acsincome_preprocessed.csv")
    Y = df["PINCP"]
    X = df.drop(columns=["PINCP"])
    for col in X.columns:
        if col in CAT_FEATURES["acsincome"]:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X, Y


def preprocess_dataset(
    dataset: str, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    X_train = preprocess.transform(X_train)  # type: ignore
    X_val = preprocess.transform(X_val)  # type: ignore
    X_test = preprocess.transform(X_test)  # type: ignore
    return X_train, X_val, X_test


def get_subgroup_feature(dataset: str, X: pd.DataFrame, n_groups: int = 2) -> pd.Series:
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
        elif dataset == "enem" or dataset == "enemreg":
            A = X.racebin.astype(str)

    elif n_groups == 4:
        if dataset == "german":
            A = (
                X.Gender.astype(str)
                + ","
                + (X.Age > 30).apply(lambda x: "Older 30" if x else "Under 30")
            )
        elif dataset == "compas":

            def race_cat(race):
                if race in ["African-American", "Caucasian", "Hispanic"]:
                    return race
                else:
                    return "Other"

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
        elif dataset == "enem" or dataset == "enemreg":

            def race_cat(race):
                if race in ["White"]:
                    return "1"
                else:
                    return "0"

            A = X.racebin.apply(race_cat).astype(str) + "_" + X.sexbin.astype(str)

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
        elif dataset == "enem" or dataset == "enemreg":

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
        elif dataset == "enem" or dataset == "enemreg":

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
        assert dataset == "enemlarge"
        sg_columns = [col for col in X.columns if "SG_UF_PROVA" in col]
        # A is the index of the non zero column
        A = X[sg_columns].idxmax(axis=1)

    sensitive_map = dict([(attr, i) for i, attr in enumerate(A.unique())])
    print(sensitive_map)
    A = A.map(sensitive_map)
    return A


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
