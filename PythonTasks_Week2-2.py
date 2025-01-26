import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set pandas display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Read the dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\persona.csv")

# TASK 1 - General Information of the Dataset

# 1-1. Display general information about the dataset
print(df.info())

# TASK 1 - Data Analysis

# 1-2. How many unique SOURCES are there? What are their frequencies?
print(df["SOURCE"].value_counts())

# Visualize the frequencies of SOURCE
sns.countplot(x="SOURCE", data=df)
plt.title("Source Frequency")
plt.show()

# 1-3. How many unique PRICES are there?
print(df["PRICE"].nunique())

# 1-4. How many sales have occurred for each PRICE?
print(df["PRICE"].value_counts())

# 1-5. How many sales have occurred for each COUNTRY?
print(df.groupby("COUNTRY")["PRICE"].count())

# 1-6. How much total revenue has been earned from each COUNTRY?
print(df.groupby("COUNTRY")["PRICE"].sum())

# 1-7. What are the sales counts by SOURCE type?
print(df.groupby("SOURCE")["PRICE"].count())

# 1-8. What are the average PRICE values by COUNTRY?
print(df.groupby("COUNTRY")["PRICE"].mean())

# 1-9. What are the average PRICE values by SOURCE?
print(df.groupby("SOURCE")["PRICE"].mean())

# 1-10. What are the average PRICE values by COUNTRY-SOURCE combination?
print(df.pivot_table("PRICE", "COUNTRY", "SOURCE", aggfunc="mean"))

# TASK 2 - Average earnings by COUNTRY, SOURCE, SEX, and AGE

print(df.pivot_table("PRICE", ["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean"))

# TASK 3 - Sort the output by PRICE

agg_df = df.pivot_table("PRICE", ["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean").sort_values(by="PRICE", ascending=True)

print(agg_df)

# TASK 4 - Convert index names into variables

agg_df = agg_df.reset_index()

# TASK 5 - Convert the AGE variable to a categorical variable and add it to agg_df

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0,15,25,35,50,70], labels=["child", "teen", "young adult", "adult", "elderly"])

print(agg_df)

# TASK 6 - Define new persona-based customers

agg_df["customers_level_based"] = ["_".join([str(row[col]) for col in ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]])  for _, row in agg_df.iterrows()]

print(agg_df.nunique())

print(agg_df.groupby("customers_level_based")["PRICE"].mean())

# TASK 7 - Segment the new customers (personas)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, ["A", "B", "C", "D"])

print(agg_df.groupby("SEGMENT").agg({"PRICE": ["sum", "mean", "min", "max"]}))

# TASK 8 - Classify new customers and predict how much revenue they might generate

new_user = "tur_android_female_young adult"
new_user2 = "fra_ios_female_young adult"

print(agg_df[agg_df["customers_level_based"] == new_user].describe().T)

print(agg_df[agg_df["customers_level_based"] == new_user2])

# I will improve this answer#
