import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importing categorical column detection function (if available in another module)
from EDA.CVandGeneralizingOP import cat_cols

# Setting Pandas display options for better visibility
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Loading the Titanic dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\feature_engineering\datasets\titanic.csv")

# Initializing LabelEncoder
le = LabelEncoder()
le.fit_transform(df["Sex"])[:5]  # Encoding the 'Sex' column
le.inverse_transform([0,1])  # Decoding the transformed values

# Function for binary label encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Encoding binary categorical columns
label_encoder(df, "Sex")

# Identifying binary categorical columns
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]

# Applying label encoding to all binary categorical columns
for col in binary_cols:
    label_encoder(df, col)

df  # Displaying the updated dataframe

# ---------------- One-Hot Encoding ---------------- #

# Different ways to apply One-Hot Encoding
pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=np.uint8).head()

# Checking dataset info
df.info()

# Function for One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Identifying categorical columns with more than 2 unique values but less than or equal to 10
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

# Applying One-Hot Encoding
one_hot_encoder(df, ohe_cols)
