import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Pandas display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load and preprocess the breast cancer dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\breast_cancer.csv")
df = df.iloc[:, 2:-1]

# Identify numerical columns
num_col = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]

# Compute correlation matrix
correlate = df[num_col].corr()

# Plot the heatmap of correlations
sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(correlate, cmap="RdBu")
plt.title("Correlation Heatmap")
plt.show()

# Calculate the absolute correlation matrix
cor_matrix_ex = df.corr().abs()

# Extract the upper triangle of the correlation matrix
upper_triangle_matrix_ex = cor_matrix_ex.where(np.triu(np.ones(cor_matrix_ex.shape), k=1).astype(bool))

# Identify columns to drop based on high correlation
threshold = 0.90
drop_list1 = [col for col in upper_triangle_matrix_ex.columns if any(upper_triangle_matrix_ex[col] > threshold)]

# Function to identify and optionally plot highly correlated columns
def high_corr_cols(dataframe, plot=False, corr_th=0.90):
    """
    Identifies columns with high correlation in a dataframe and optionally plots the correlation heatmap.

    Parameters:
    dataframe (pd.DataFrame): The dataframe to analyze.
    plot (bool): Whether to plot the correlation heatmap. Default is False.
    corr_th (float): Correlation threshold above which columns are considered highly correlated. Default is 0.90.

    Returns:
    list: List of columns to drop due to high correlation.
    """
    corr = dataframe.corr()
    cor_matrix = dataframe.corr().abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.title("Correlation Heatmap with Threshold")
        plt.show()
    return drop_list

# Example usage:
# drop_list = high_corr_cols(df, plot=True)
