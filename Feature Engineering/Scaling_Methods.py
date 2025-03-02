import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from EDA.EDA_Num import num_summary

# Display all columns and adjust output width for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the Titanic dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\feature_engineering\datasets\titanic.csv")

# StandardScaler (SS) is sensitive to outliers
ss = StandardScaler()
df["Age_ss"] = ss.fit_transform(df[["Age"]])

# RobustScaler (RS) is based on the median and interquartile range (IQR), making it more robust to outliers
rs = RobustScaler()
df["Age_rs"] = rs.fit_transform(df[["Age"]])

# Display summary statistics
df.describe().T

# MinMaxScaler (MMS) scales values between a given range (default: 0 to 1)
mms = MinMaxScaler()
df["Age_mms"] = mms.fit_transform(df[["Age"]])

# Display summary statistics after scaling
df.describe().T

# Extract all columns that contain "Age" for numerical summary
age_cols = [col for col in df.columns if "Age" in col]
num_summary(df, age_cols)

# Numeric to Categorical Transformation using binning (qcut)
# Example implementation can be added if required
