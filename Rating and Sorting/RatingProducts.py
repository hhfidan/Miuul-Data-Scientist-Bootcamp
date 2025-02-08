import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Display settings for Pandas DataFrame
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Load dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\measurement_problems\datasets\course_reviews.csv")

# Check for missing values
print(df.isnull().sum())

# Count occurrences of each rating
df["Rating"].value_counts()

# Calculate the mean rating based on the number of questions asked
df.groupby("Questions Asked")["Rating"].mean()
df.groupby("Questions Asked").agg({"Questions Asked": "count", "Rating": "mean"})

# Convert Timestamp column to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Define a reference date
today_date = pd.to_datetime("2021-02-10 0:0:0")

# Calculate the number of days since the review was given
df["days"] = (today_date - df["Timestamp"]).dt.days

# Calculate mean ratings based on time intervals
df.loc[df["days"] <= 30, "Rating"].mean()
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[df["days"] > 180, "Rating"].mean()

# Compute time-based weighted average rating
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

time_based_weighted_average(df, 30, 26, 22, 22)

# Compute user-based weighted average rating
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["Progress"] > 75, "Rating"].mean() * w4 / 100

user_based_weighted_average(df)

# Compute overall course weighted rating
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)
