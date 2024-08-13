import os
from copy import deepcopy
import numpy as np
import pandas as pd


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
        "DEFAULT",
    ]
    df["DEFAULT"] = df.DEFAULT.apply(lambda x: 1 if x == 2 else 0)
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


def preprocess_compas():
    df = pd.read_csv("data/compas-scores-two-years.csv")
    df = df[
        [
            "sex",
            "age_cat",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "two_year_recid",
        ]
    ]
    df = df.dropna()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.Categorical(df[col])

    df.to_csv("data/compas_preprocessed.csv", index=False)


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


def preprocess_taiwan():
    df = pd.read_csv("data/taiwan.csv")
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
    df.to_csv("data/taiwan_preprocessed.csv", index=False)


def preprocess_enem(
    grade_attribute=["NU_NOTA_CH"],
    n_sample=50000,
    n_sample_large=1000000,
    multigroup=True,
    n_classes=2,
):
    from sklearn.preprocessing import MinMaxScaler

    features = (
        ["NU_NOTA_CH", "TP_COR_RACA", "TP_SEXO"]
        + ["Q00" + str(x) if x < 10 else "Q0" + str(x) for x in range(1, 25)]
        + ["SG_UF_PROVA", "TP_FAIXA_ETARIA"]
    )

    def construct_race(df, protected_attribute):
        race_dict = {
            "Branca": "White",
            "Preta": "Black",
            "Parda": "Brown",
            "Amarela": "Asian",
            "Indigena": "Native",
        }  # changed to match ENEM 2020 numbering
        return df[protected_attribute].map(race_dict)

    def construct_grade(df, grade_attribute, n):
        v = df[grade_attribute[0]].values
        quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n + 1))
        return pd.cut(v, quantiles, labels=np.arange(n))

    df = pd.read_csv("../data/MICRODADOS_ENEM_2020.csv", encoding="cp860", sep=";")

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = (
        ~df[["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]]
        .applymap(lambda x: False if x == 1.0 else True)
        .any(axis=1)
    )
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df["IN_TREINEIRO"] == 0, :]

    ## drop eliminated features
    df.drop(
        [
            "TP_PRESENCA_CN",
            "TP_PRESENCA_CH",
            "TP_PRESENCA_LC",
            "TP_PRESENCA_MT",
            "IN_TREINEIRO",
        ],
        axis=1,
        inplace=True,
    )

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, "Branca", "Preta", "Parda", "Amarela", "Indigena"]
    df["TP_COR_RACA"] = (
        df.loc[:, ["TP_COR_RACA"]].applymap(lambda x: race_names[x]).copy()
    )

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])]

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df["gradebin"] = construct_grade(df, grade_attribute, n_classes)
    df["gradescore"] = df[grade_attribute[0]]
    if multigroup:
        df["racebin"] = construct_race(df, "TP_COR_RACA")
    else:
        df["racebin"] = np.logical_or(
            (df["TP_COR_RACA"] == "Branca").values,
            (df["TP_COR_RACA"] == "Amarela").values,
        ).astype(int)
    df["sexbin"] = df["TP_SEXO"].apply(lambda x : 1 if x == "M" else 0)

    df.drop([grade_attribute[0], "TP_COR_RACA", "TP_SEXO"], axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ["Q00" + str(x) if x < 10 else "Q0" + str(x) for x in range(1, 25)]
    for q in question_vars:
        if q != "Q005":
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if "TP_FAIXA_ETARIA" in features:
        q = "TP_FAIXA_ETARIA"
        df_q = pd.get_dummies(df[q], prefix=q)
        df.drop([q], axis=1, inplace=True)
        df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_PROVA (state where exam was taken)
    df_res = pd.get_dummies(df["SG_UF_PROVA"], prefix="SG_UF_PROVA")
    df.drop(["SG_UF_PROVA"], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    df = df.dropna()
    ## Scaling ##
    scaler = MinMaxScaler()
    scale_columns = list(set(df.columns.values) - set(["gradebin", "racebin"]))
    df[scale_columns] = pd.DataFrame(
        scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index
    )
    # print('Preprocessed Dataset Shape:', df.shape)

    df_large = df.sample(n=min(n_sample_large, df.shape[0]), axis=0, replace=False)
    df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)

    df_large = df_large.drop(["gradescore"], axis=1)
    df_classif = df.drop(["gradescore"], axis=1)
    df_reg = df.drop(["gradebin"], axis=1)

    df_large.to_csv("../data/enem_large_preprocessed.csv", index=False)
    df_classif.to_csv("../data/enem_classif_preprocessed.csv", index=False)
    df_reg.to_csv("../data/enem_reg_preprocessed.csv", index=False)


def download_data():
    import gdown

    url = "https://drive.google.com/drive/folders/12r-AU7HS9XBcfv__9aUYvwEZDsAbcOwK"
    gdown.download_folder(url, quiet=True, use_cookies=False)


if __name__ == "__main__":
    download_data()
