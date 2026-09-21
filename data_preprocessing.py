import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv("train.csv")
data["GroupId"] = data["PassengerId"].str.split("_", expand=True)[0]
data["GroupSize"] = data.groupby("GroupId")["GroupId"].transform("size")
data = data.drop(["GroupId"], axis=1)
data = data.set_index("PassengerId")

y = data["Transported"]
X = data.drop(["Transported"], axis=1)

X = X.drop(["Name"], axis=1)
X[["Cabin deck", "Cabin number", "Cabin side"]] = X["Cabin"].str.split("/", expand=True)
X = X.drop(["Cabin"], axis=1)

X["Cabin number"] = X["Cabin number"].astype(float)
X["CryoSleep"] = X["CryoSleep"].astype(float)
X["VIP"] = X["VIP"].astype(float)


num_cols = [col for col in X.select_dtypes(include=["number","bool"])]
high_cat_cols = [col for col in X.select_dtypes(exclude=["number", "bool"]) if X[col].nunique() >= 10]
low_cat_cols = [col for col in X.select_dtypes(exclude=["number", "bool"]) if X[col].nunique() < 10]

preprocessing1 = ColumnTransformer([('Num processing', SimpleImputer(strategy="mean"), num_cols), 
                                   ('Low category processing', Pipeline([("1", SimpleImputer(strategy="most_frequent")),
                                                            ("2", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low_cat_cols), 
                                    ('High category processing', Pipeline([("1", SimpleImputer(strategy="most_frequent")), 
                                                            ("2", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high_cat_cols)])

preprocessing2 = ColumnTransformer([('Num processing', SimpleImputer(strategy="mean"), num_cols), 
                                   ('Low category processing', Pipeline([("1", SimpleImputer(strategy="constant")),
                                                            ("2", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low_cat_cols), 
                                    ('High category processing', Pipeline([("1", SimpleImputer(strategy="constant")), 
                                                            ("2", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high_cat_cols)])

preprocessing3 = ColumnTransformer([('Num processing', SimpleImputer(strategy="constant"), num_cols), 
                                   ('Low category processing', Pipeline([("1", SimpleImputer(strategy="most_frequent")),
                                                            ("2", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low_cat_cols), 
                                    ('High category processing', Pipeline([("1", SimpleImputer(strategy="most_frequent")), 
                                                            ("2", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high_cat_cols)])

preprocessing4 = ColumnTransformer([('Num processing', SimpleImputer(strategy="constant"), num_cols), 
                                   ('Low category processing', Pipeline([("1", SimpleImputer(strategy="constant")),
                                                            ("2", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low_cat_cols), 
                                    ('High category processing', Pipeline([("1", SimpleImputer(strategy="constant")), 
                                                            ("2", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high_cat_cols)])
