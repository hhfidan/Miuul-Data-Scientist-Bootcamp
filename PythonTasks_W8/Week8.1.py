import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from click import style
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, RandomizedSearchCV, KFold
from skompiler import skompile


# Set options to display full dataframe without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore", category=Warning)

########################################################
########## Task 1: Exploratory Data Analysis ###############

# Step 1: Read train and test datasets, and combine them into one dataframe
df_test = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week8\test.csv")
df_train = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week8\train.csv")

df = pd.concat([df_train, df_test], axis=0)

# Step 2: Identify numeric and categorical variables
##### Functions taken from week 7-3
checker(df)
categoric, numericcol, cardinal, blank_str, num_with_zero = grab(df)

# Step 3: Perform necessary data type corrections (e.g., incorrect types)
for col in categoric:
    df[col] = df[col].astype(str)

df["OverallCond"] = df["OverallCond"].astype(int)
df["OverallQual"] = df["OverallQual"].astype(int)
df["PoolArea"] = df["PoolArea"].astype(int)

############################################################
# Step 4: Visualize distribution of numeric and categorical variables
cat_num_summary(df, categoric, numericcol)

# Display missing values count and details
df.isnull().sum().sort_values(ascending=False)

# Display rows with missing values
df[df.isnull().any(axis=1)]
df[df.columns[df.isnull().any()]].head()

# Correlation heatmap between 'MSSubClass' and 'SalePrice'
corr = df[['MSSubClass', 'SalePrice']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("HouseStyle & SalePrice Correlation")
plt.show()

##############################################
## Step 5: Analyze categorical variables and target variable

def cat_summary(dataframe, col_name, target):
    print(pd.DataFrame({
        "Count": df[col_name].value_counts(),
        "Ratio": df[col_name].value_counts() / len(df),
        "Mean": df.groupby(col_name)[target].mean()
    }))
    print("--------------------------------------------------")

for col in categoric:
    cat_summary(df, col, "SalePrice")

###################################################################

# Step 6: Check for outliers in numeric variables

numericcol.remove("SalePrice")

for col in numericcol:
    print(col, checkOutlier(df, col))

# Apply Local Outlier Factor to detect outliers
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(df[numericcol])
df_scores = lof.negative_outlier_factor_

# Plot outlier scores
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style=".-")
plt.show()

# Count outliers with scores less than a threshold
print(len(df_scores[df_scores < -1.5]))

# Apply threshold for outlier removal
th = np.sort(df_scores)[48]
th = np.percentile(df_scores, 5)
df = df[df_scores > th]

# Step 7: Analyze missing values and handle them

sns.histplot(df['MasVnrType'], kde=True, bins=30, color='blue')
plt.title('MasVnrType Distribution (Histogram + KDE)')
plt.xlabel('MasVnrType')
plt.ylabel('Frequency / Density')
plt.show()

# Fill missing values with median or mode as appropriate
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
df["LotFrontage"] = df["LotFrontage"].fillna(df.groupby("LotConfig")["LotFrontage"].transform("median"))
df["MasVnrArea"] = df["MasVnrArea"].fillna(df.groupby("MasVnrType")["MasVnrArea"].transform("median"))
df[["BsmtUnfSF", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF"]] = df[["BsmtUnfSF", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF"]].fillna(0)
df["GarageArea"] = df["GarageArea"].fillna(0)
df["Exterior1st"] = df["Exterior1st"].fillna(df.groupby("YearBuilt")["Exterior1st"].transform(lambda x: x.mode()[0]))
df["Exterior2nd"] = df["Exterior2nd"].fillna(df.groupby("YearBuilt")["Exterior2nd"].transform(lambda x: x.mode()[0]))

# Check remaining missing values
df.isnull().sum().sort_values(ascending=False)

# Display rows with missing values
df[df.isnull().any(axis=1)]
df[df.columns[df.isnull().any()]].head()

# Final check for missing values in 'Exterior1st'
df[df["Exterior1st"].isnull()]
df[df["Exterior1st"] == "nan"]

# Additional observations for missing data
# PoolQC 2909, MiscFeature 2814, Alley 2721, etc.

# Correlation heatmap for categorical variables
df[categoric].corr()
corr = df[numericcol].corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[categoric].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
