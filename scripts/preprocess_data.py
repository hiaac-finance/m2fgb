import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_taiwan():
    df = pd.read_csv("data/Taiwan.csv")
    df.columns = df.iloc[0, :].tolist()
    df = df.iloc[1:, :]
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": "DEFAULT"})
    df = df.astype("float64")
    sex_map = {2: "Female", 1: "Male"}
    education_map = {
        -2: "Unknown",
        -1: "Unknown",
        0: "Unknown",
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others",
        5: "Unknown",
        6: "Unknown",
    }
    marriage_map = {
        0: "Others",
        1: "Married",
        2: "Single",
        3: "Others",
    }
    df.SEX = df.SEX.apply(sex_map.get)
    df.EDUCATION = df.EDUCATION.apply(education_map.get)
    df.MARRIAGE = df.MARRIAGE.apply(marriage_map.get)
    cat_cols = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    for col in df.columns:
        if col in cat_cols or col == "DEFAULT":
            continue
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])

    df=pd.get_dummies(df,columns=cat_cols,prefix=cat_cols)

    df.to_csv("data/taiwan_preprocessed.csv", index = False)
    

def preprocess_german():
    df = pd.read_csv("data/german.csv", header=None)
    df.columns = [
        "CheckingAccount",
        "Duration",
        "CreditHistory",
        "Purpose",
        "CreditAmount",
        "SavingsAccount",
        "EmploymentSince",
        "InstallmentRate",
        "PersonalStatus",
        "OtherDebtors",
        "ResidenceSince",
        "Property",
        "Age",
        "OtherInstallmentPlans",
        "Housing",
        "ExistingCredits",
        "Job",
        "Dependents",
        "Telephone",
        "ForeignWorker",
        "DEFAULT"
    ]
    df["DEFAULT"] = df.DEFAULT.apply(lambda x : 1 if x == 2 else 0)
    df["Gender"] = df.PersonalStatus
    df.CheckingAccount = df.CheckingAccount.apply(
        {"A11": "< 0", "A12": "0 - 200", "A13": "> 200", "A14": "No"}.get
    )
    df.CreditHistory = df.CreditHistory.apply(
        {
            "A30": "No credits/all paid",
            "A31": "All paid",
            "A32": "Existing paid",
            "A33": "Delay in paying",
            "A34": "Critical account",
        }.get
    )
    df.Purpose = df.Purpose.apply(
        {
            "A40": "Car (new)",
            "A41": "Car (used)",
            "A42": "Furniture/equipment",
            "A43": "Radio/television",
            "A44": "Domestic appliances",
            "A45": "Repairs",
            "A46": "Education",
            "A47": "Vacation",
            "A48": "Retraining",
            "A49": "Business",
            "A410": "Others",
        }.get
    )
    df.SavingsAccount = df.SavingsAccount.apply(
        {
            "A61": "< 100",
            "A62": "100 - 500",
            "A63": "500 - 1000",
            "A64": "> 1000",
            "A65": "Unknown/None",
        }.get
    )
    df.EmploymentSince = df.EmploymentSince.apply(
        {
            "A71": "Unemployed",
            "A72": "< 1",
            "A73": "1 - 4",
            "A74": "4 - 7",
            "A75": "> 7",
        }.get
    )
    df.Gender = df.Gender.apply(
        {
            "A91": "Male",
            "A92": "Female",
            "A93": "Male",
            "A94": "Male",
            "A95": "Female",
        }.get
    )
    df.OtherDebtors = df.OtherDebtors.apply(
        {"A101": "No", "A102": "Co-applicant", "A103": "Guarantor"}.get
    )
    df.Property = df.Property.apply(
        {
            "A121": "Real estate",
            "A122": "Savings agreement/life insurance",
            "A123": "Car or other",
            "A124": "Unknown/None",
        }.get
    )
    df.OtherInstallmentPlans = df.OtherInstallmentPlans.apply(
        {"A141": "Bank", "A142": "Stores", "A143": "No"}.get
    )
    df.Housing = df.Housing.apply(
        {"A151": "Rent", "A152": "Own", "A153": "For free"}.get
    )
    df.Job = df.Job.apply(
        {
            "A171": "Unemployed",
            "A172": "Unskilled",
            "A173": "Skilled",
            "A174": "Highly skilled",
        }.get
    )
    df.Telephone = df.Telephone.apply({"A191": 0, "A192": 1}.get)
    df.ForeignWorker = df.ForeignWorker.apply({"A201": 1, "A202": 0}.get)
    df = df.drop(columns=["PersonalStatus"])
    df["DEFAULT"] = 1 - df["DEFAULT"]
    df = df.rename(columns={"DEFAULT": "GOOD_RISK"})
    df.to_csv("data/german_preprocessed.csv", index=False)

def preprocess_adult():
    from ucimlrepo import fetch_ucirepo
    adult = fetch_ucirepo(id=2)
    df = adult.data.features.copy()
    df["INCOME"] = adult.data.targets
    df = df.drop(columns=["fnlwgt"])
    df.to_csv("data/adult_preprocessed.csv", index=False)

def download_data():
    import gdown
    url = "https://drive.google.com/drive/folders/12r-AU7HS9XBcfv__9aUYvwEZDsAbcOwK"
    gdown.download_folder(url, quiet=True, use_cookies=False)

if __name__ == "__main__":
    download_data()