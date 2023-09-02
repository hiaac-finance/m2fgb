import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_german():
    df = pd.read_csv("../data/german_credit_data_K.csv")
    df=df.iloc[:,1:]
    df["Monthly pay"] = (df["Credit amount"] / df["Duration"])
    df["Credit amount^2"] = df["Credit amount"]**2
    df.insert(1,"Cat Age", np.NaN)
    df.loc[df["Age"]<25,"Cat Age"]="0-25"
    df.loc[((df["Age"]>=25) & (df["Age"]<30)),"Cat Age"]="25-30"
    df.loc[((df["Age"]>=30) & (df["Age"]<35)),"Cat Age"]="30-35"
    df.loc[((df["Age"]>=35) & (df["Age"]<40)),"Cat Age"]="35-40"
    df.loc[((df["Age"]>=40) & (df["Age"]<50)),"Cat Age"]="40-50"
    df.loc[((df["Age"]>=50) & (df["Age"]<76)),"Cat Age"]="50-75"
    df.insert(9,"Cat Duration",df["Duration"])
    for i in df["Cat Duration"]:
        if i<12:
            df["Cat Duration"]=df["Cat Duration"].replace(i,"0-12")
        elif (i>=12) and (i<24):
            df["Cat Duration"]=df["Cat Duration"].replace(i,"12-24")
        elif (i>=24) and (i<36):
            df["Cat Duration"]=df["Cat Duration"].replace(i,"24-36")
        elif (i>=36) and (i<48):
            df["Cat Duration"]=df["Cat Duration"].replace(i,"36-48")
        elif (i>=48) and (i<60):
            df["Cat Duration"]=df["Cat Duration"].replace(i,"48-60")
        elif (i>=60) and (i<=72):
            df["Cat Duration"]=df["Cat Duration"].replace(i,"60-72")
    df.insert(4,"Cat Job",df["Job"])
    df["Cat Job"]=df["Cat Job"].astype("category")
    df["Cat Job"]=df["Cat Job"].replace(0,"unskilled")
    df["Cat Job"]=df["Cat Job"].replace(1,"resident")
    df["Cat Job"]=df["Cat Job"].replace(2,"skilled")
    df["Cat Job"]=df["Cat Job"].replace(3,"highly skilled")
    df["Job"]=pd.Categorical(df["Job"],categories=[0,1,2,3],ordered=True)
    df["Cat Age"]=pd.Categorical(df["Cat Age"],categories=['0-25','25-30', '30-35','35-40','40-50','50-75'])
    df["Cat Duration"]=pd.Categorical(df["Cat Duration"],categories=['0-12','12-24', '24-36','36-48','48-60','60-72'])
    df["Age"],df["Duration"],df["Job"]=df["Cat Age"],df["Cat Duration"],df["Cat Job"]
    df=df.drop(["Cat Age","Cat Duration","Cat Job"],axis=1)
    # "female" replaced to 1
    df["Sex"] = df["Sex"].apply(lambda x : 1 if x == "female" else 0).astype(int)
    # "good" replaced to 1
    df["Risk"] = df["Risk"].apply(lambda x : 1 if x == "good" else 0)
    one_hot_columns=list(df.columns)
    one_hot_columns.remove("Sex")
    one_hot_columns.remove("Risk")
    one_hot_columns.remove("Credit amount")
    one_hot_columns.remove("Monthly pay")
    one_hot_columns.remove("Credit amount^2")
    df=pd.get_dummies(df,columns=one_hot_columns,prefix=one_hot_columns)
    scaler=MinMaxScaler()
    df["Credit amount"]=scaler.fit_transform(df[["Credit amount"]])
    df["Monthly pay"]=scaler.fit_transform(df[["Monthly pay"]])
    df["Credit amount^2"]=scaler.fit_transform(df[["Credit amount^2"]])
    df.to_csv("../data/german_credit_data_K_preprocessed.csv", index = False)

if __name__ == "__main__":
    preprocess_german()