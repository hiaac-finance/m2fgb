import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def load_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    X = df.drop(["Risk"], axis=1)
    Y = df["Risk"]
    return X, Y


def load_taiwan():
    df = pd.read_csv("../data/taiwan_preprocessed.csv")
    X = df.drop(["DEFAULT"], axis=1)
    Y = df["DEFAULT"]
    X.insert(0, "Sex", X["SEX_Female"])
    X = X.drop(columns=["SEX_Female", "SEX_Male"])
    return X, Y


def load_german2():
    df = pd.read_csv("../data/german_preprocessed.csv")
    X = df.drop(["DEFAULT"], axis=1)
    Y = df["DEFAULT"]
    return X, Y


def load_dataset(dataset):
    if dataset == "german":
        return load_german()
    elif dataset == "taiwan":
        return load_taiwan()
    elif dataset == "german2":
        return load_german2()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def load_unbalanced_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    df = pd.concat([df[df.Sex == 0], df[df.Sex == 1].sample(70)]).reset_index(drop=True)
    X = df.drop(["Risk"], axis=1)
    Y = df["Risk"]
    return X, Y


def get_fold(dataset, fold, random_state=None):
    X, Y = load_dataset(dataset)
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i == fold:
            X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=1 / 9, random_state=random_state
            )
            X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
