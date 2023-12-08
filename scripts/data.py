import pandas as pd

def load_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    X=df.drop(["Risk"],axis=1)
    Y=df["Risk"]
    return X, Y

def load_taiwan():
    df = pd.read_csv("../data/taiwan_preprocessed.csv")
    X = df.drop(["DEFAULT"], axis = 1)
    Y = df["DEFAULT"]
    X.insert(0, "Sex", X["SEX_Female"])
    X = X.drop(columns = ["SEX_Female", "SEX_Male"])
    return X, Y

def load_dataset(dataset):
    if dataset == "german":
        return load_german()
    elif dataset == "taiwan":
        return load_taiwan()
    else:
        raise ValueError(f"Unknown dataset {dataset}")

def load_unbalanced_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    df = pd.concat([
        df[df.Sex == 0],
        df[df.Sex == 1].sample(70)
    ]).reset_index(drop = True)
    X=df.drop(["Risk"],axis=1)
    Y=df["Risk"]
    return X, Y