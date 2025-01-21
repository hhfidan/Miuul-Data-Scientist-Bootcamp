import numpy as np
import pandas as pd
import seaborn as sns
from distributed.diagnostics.progress_stream import counts

# --- NumPy Examples ---
# Create an array of zeros with 10 elements
a = np.zeros(10, dtype=int)

# Create a NumPy array and demonstrate slicing
b = np.array([1, 2, 34, 5, 6])
a_random = np.random.randint(0, 10, size=10)
a_normal = np.random.normal(10, 2, (3, 4))
reshaped = a_normal.reshape(4, 3)

# Indexing and filtering with NumPy
catch = [1, 2, 3]
selected = b[catch]
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

array = np.array([1, 2, 3, 4, 5, 6, 7])
filter_array = array % 2 == 0
new_array = array[filter_array]
print(new_array)

# --- Pandas Examples ---
# Load Titanic dataset using Seaborn
df = sns.load_dataset("titanic")

# Basic operations on DataFrame
df.head()
missing_values = df.isnull().values.any()
sex_counts = df["sex"].value_counts()

# Manipulating index and columns
df.index = df["age"]
df.drop("age", axis=1, inplace=True)
df.reset_index(inplace=True)

# Filtering DataFrame
df[df["age"] > 50]["age"].count()
df.loc[df["age"] > 70, ["age", "sex"]]
df.loc[(df["age"] > 70) & (df["sex"] == "male"), ["age", "alive"]]

# Grouping and aggregation
pd.set_option('display.width', 500)
age_sex_grouped = df.groupby("sex").agg({"age": ["mean", "sum"]})
multilevel_grouping = (df.groupby(["sex", "embark_town", "class"]) 
                       .agg({"age": ["mean"], "survived": ["mean"], "sex": ["count"]}))

# Pivot table
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
pivot = df.pivot_table("survived", "sex", "new_age", observed=True)

# Feature engineering
df.drop("new_age", axis=1, inplace=True)
df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 3
normalized = df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std())
df.loc[:, df.columns.str.contains("age")] = normalized

# Combining DataFrames
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 66
df3 = pd.concat([df1, df2], axis=1)

print(df3)
