import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set display options for better visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Task 1 - Load the Titanic dataset from the seaborn library
df = sns.load_dataset("titanic")

# Task 2 - Count the number of male and female passengers
print(df["sex"].value_counts())

# Task 3 - Find the number of unique values for each column
print(df.nunique())

# Task 4 - Find the unique values in the pclass column
print(df["pclass"].unique())

# Task 5 - Find the unique values in the pclass and parch columns
print(df[["pclass", "parch"]].nunique())

# Task 6 - Check the data type of the embarked column, convert it to category, and check again
print(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)

# Task 7 - Show all information for passengers with embarked value "C"
print(df[df["embarked"] == "C"])

# Task 8 - Show all information for passengers whose embarked value is not "S"
print(df[df["embarked"] != "S"])

# Task 9 - Show all information for passengers under 30 years old who are female
print(df.loc[(df["age"] < 30) & (df["sex"] == "female")])

# Task 10 - Show information for passengers with fare > 500 or age > 70
print(df.loc[(df["fare"] > 500) | (df["age"] > 70)])

# Task 11 - Find the total number of missing values for each column
print(df.isnull().sum())

# Task 12 - Drop the "who" column from the dataframe
df.drop("who", axis=1, inplace=True)

# Task 13 - Fill missing values in the "deck" column with its mode
mode_value = df["deck"].mode()[0]
df["deck"].fillna(mode_value, inplace=True)

# Task 14 - Fill missing values in the "age" column with its median
median_value = df["age"].median()
df["age"].fillna(median_value, inplace=True)

# Task 15 - Find sum, count, and mean of survived grouped by pclass and sex
print(df.pivot_table("survived", index="sex", columns="pclass", aggfunc=["mean", "sum", "count"]))

# Task 16 - Create an age_flag column: 1 for age < 30, 0 otherwise
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# Task 17 - Load the Tips dataset from seaborn library
df = sns.load_dataset("tips")

# Task 18 - Find total, min, max, and mean of total_bill grouped by time
print(df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]}))

# Task 19 - Find total, min, max, and mean of total_bill grouped by day and time
print(df.pivot_table("total_bill", index="day", columns="time", aggfunc=["sum", "min", "max", "mean"]))

# Task 20 - Find total, min, max, and mean of total_bill and tip for female customers during Lunch, grouped by day
print(df[(df["time"] == "Lunch") & (df["sex"] == "Female")].pivot_table(
    index="day",
    values=["total_bill", "tip"],
    aggfunc=["sum", "min", "max", "mean"]
))

# Task 21 - Find the mean of orders where size < 3 and total_bill > 10
print(df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].select_dtypes(include=["number"]).mean())

# Task 22 - Create a new column total_bill_tip_sum as the sum of total_bill and tip
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# Task 23 - Sort by total_bill_tip_sum in descending order and save the top 30 rows to a new dataframe
df_top30 = df.sort_values(by="total_bill_tip_sum", ascending=False).head(30)
print(df_top30)
