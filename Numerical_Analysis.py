import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Pandas display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Identify numerical columns with more than 10 unique values
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and df[col].nunique() > 10]

# Define a function to summarize numerical columns
def num_summary(dataframe, numeric_col):
    """
    Prints summary statistics and plots a histogram for a numerical column.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the data.
    numeric_col (str): The name of the numerical column to summarize.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90, 0.95, 1]
    print(dataframe[numeric_col].describe(quantiles).T)
    dataframe[numeric_col].hist()
    plt.xlabel(numeric_col)
    plt.title(f"Histogram of {numeric_col}")
    plt.show()

# Apply the summary function to each numerical column
for col in num_cols:
    num_summary(df, col)
