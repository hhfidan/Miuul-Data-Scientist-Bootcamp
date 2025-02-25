import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from matplotlib import pyplot as plt
from dsbootcamp.prep.outlier import check_outlier  # Custom function for detecting outliers

# Pandas display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Load the 'diamonds' dataset from Seaborn
df = sns.load_dataset("diamonds")

# Select only numerical columns
df = df.select_dtypes(include=["float64", "int64"])

# Drop any missing values
df = df.dropna()

# Display the first few rows
df.head()

# Check for outliers in each numerical column
for col in df.columns:
    print(col, check_outlier(df, col))

# Apply Local Outlier Factor (LOF) for anomaly detection
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

# Obtain the negative outlier factor scores
df_scores = clf.negative_outlier_factor_

# Display sorted outlier scores
np.sort(df_scores)[0:5]

# Plot the sorted outlier scores
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

# Set a threshold for outlier detection
th = np.sort(df_scores)[3]

# Display potential outliers
df[df_scores < th]

# Remove detected outliers from the dataset
df_cleaned = df[df_scores >= th]

# Display the cleaned dataset
df_cleaned.head()
