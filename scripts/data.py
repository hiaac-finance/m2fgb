import pandas as pd

def load_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    X=df.drop(["Risk"],axis=1)
    Y=df["Risk"]
    return X, Y

def load_unbalanced_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    df = pd.concat([
        df[df.Sex == 0],
        df[df.Sex == 1].sample(100)
    ]).reset_index(drop = True)
    X=df.drop(["Risk"],axis=1)
    Y=df["Risk"]
    return X, Y