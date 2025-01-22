import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Pandas display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load Titanic dataset using Seaborn's built-in dataset loader
df = sns.load_dataset("titanic")

# Function to provide a comprehensive overview of the dataset
def check_df(dataframe, head=5):
    """Provides an overview of the dataframe including shape, types, missing values, and statistics."""
    print("######################## Shape ############################")
    print(dataframe.shape)
    print("######################## Types ############################")
    print(dataframe.dtypes)
    print("######################## Head ############################")
    print(dataframe.head(head))
    print("######################## Tail ############################")
    print(dataframe.tail(head))
    print("######################## NA ############################")
    print(dataframe.isnull().sum())
    print("######################## Quantiles ############################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Identify categorical columns
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "bool", "object"]]

# Identify numeric columns with a low number of unique values
num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and df[col].nunique() < 10]

# Identify high cardinality categorical columns
cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["category", "object"] and df[col].nunique() > 20]

# Function to summarize categorical columns
def cat_summary(dataframe, col_name, plot=False):
    """Displays value counts and ratios of a categorical column. Optionally plots a countplot."""
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(summary)
    print("###########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(f"Distribution of {col_name}")
        plt.show()

# Convert boolean columns to integers and generate summaries for all categorical columns
for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    cat_summary(df, col, plot=True)
