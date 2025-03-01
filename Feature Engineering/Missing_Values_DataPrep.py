import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from matplotlib import pyplot as plt

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the Titanic dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\feature_engineering\datasets\titanic.csv")
df.head()

# Function to identify missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

# Identify missing values
missing_values_table(df, True)

# Fill missing values
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
df["Cabin"] = df["Cabin"].fillna("missing")

# Verify missing values are filled
df.isnull().sum()

# Define categorical and numerical columns
from EDA.CVandGeneralizingOP import grab_col_names
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# One-hot encoding for categorical columns
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

# Impute missing values using KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# Restore imputed Age values
df["age_imputed_knn"] = dff[["Age"]]

# Identify missing values compared to target variable
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(),1,0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"Target_Mean": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n")

# Analyze missing values impact on survival
na_cols = missing_values_table(df, True)
missing_vs_target(df, "Survived", na_cols)

# Visualizing missing values
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)
plt.show()
