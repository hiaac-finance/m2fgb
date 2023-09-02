import pandas as pd

def load_german():
    df = pd.read_csv("../data/german_credit_data_K_preprocessed.csv")
    X=df.drop(["Risk"],axis=1)
    Y=df["Risk"]
    return X, Y