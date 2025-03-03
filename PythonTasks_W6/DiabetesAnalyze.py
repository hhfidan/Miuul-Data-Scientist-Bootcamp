import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week6\diabetes\diabetes.csv")

# Task 1: Exploratory Data Analysis (EDA)
# Step 1: General overview of the dataset
def check_dataframe(dataframe):
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Data Types ##################")
    print(dataframe.dtypes)
    print("################## Head and Tail ##############")
    print(dataframe.head())
    print(dataframe.tail())
    print("################## Missing Values #####################")
    print(dataframe.isnull().sum())
    print("################## Summary Statistics ##############")
    print(dataframe.describe().T)

check_dataframe(df)

# Step 2: Identifying categorical and numerical columns
def grab_columns(dataframe):
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numerical_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    high_cardinality_cols = [col for col in categorical_cols if dataframe[col].nunique() > 13]
    categorical_numeric_cols = [col for col in numerical_cols if dataframe[col].nunique() < 13]
    categorical_cols = categorical_cols + categorical_numeric_cols
    categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
    numerical_cols = [col for col in numerical_cols if col not in categorical_numeric_cols]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(categorical_cols)}")
    print(f"Numerical Columns: {len(numerical_cols)}")
    print(f"High Cardinality Categorical Columns: {len(high_cardinality_cols)}")
    print(f"Categorical Numeric Columns: {len(categorical_numeric_cols)}")
    
    return categorical_cols, numerical_cols, high_cardinality_cols

categorical_cols, numerical_cols, high_cardinality_cols = grab_columns(df)

# Step 3: Analyzing categorical and numerical variables
print(df.groupby("Outcome")[numerical_cols].mean())
print(df.groupby("Pregnancies")[categorical_cols].mean())

# Step 4: Outlier analysis
def detect_outliers(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * interquartile_range
    lower_limit = quartile1 - 1.5 * interquartile_range
    return lower_limit, upper_limit

def check_outliers(dataframe, col_name):
    lower_limit, upper_limit = detect_outliers(dataframe, col_name)
    return dataframe[(dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit)].any(axis=None)

for col in numerical_cols:
    print(col, check_outliers(df, col))

# Local Outlier Factor (LOF) for multivariate outlier detection
lof = LocalOutlierFactor(n_neighbors=20)
df_scores = lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

threshold = np.sort(df_scores)[10]
outliers = df[df_scores < threshold]
print(outliers)

# Step 5: Missing value analysis
def missing_values(dataframe):
    missing_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    missing_data = dataframe[missing_cols].isnull().sum().sort_values(ascending=False)
    missing_ratio = (dataframe[missing_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([missing_data, np.round(missing_ratio, 2)], axis=1, keys=["Missing Values", "Percentage"])
    print(missing_df)

missing_values(df)

# Step 6: Correlation analysis
correlation_matrix = df.corr()
print(correlation_matrix)
