import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Setting display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Loading the dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\measurement_problems\datasets\amazon_review.csv")

# Task 1: Calculate the Average Rating based on recent reviews and compare it with the existing average rating.

# Step 1: Calculate the overall average rating of the product.
print(df["overall"].mean())  # Since there's only one product, grouping is unnecessary.

# Step 2: Compute time-weighted average rating
def weighted_time_based_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= 100, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 100) & (dataframe["day_diff"] <= 300), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 300) & (dataframe["day_diff"] <= 600), "overall"].mean() * w3 / 100 + \
        dataframe.loc[dataframe["day_diff"] > 600, "overall"].mean() * w4 / 100

print(weighted_time_based_average(df, 30, 26, 22, 22))  # Recent reviews have a higher impact.
print(df["overall"].mean())

# Step 3: Compare and interpret the weighted average for each time period.
print(df.loc[df["day_diff"] <= 100, "overall"].mean() * 25 / 100)
print(df.loc[(df["day_diff"] > 100) & (df["day_diff"] <= 300), "overall"].mean() * 25 / 100)
print(df.loc[(df["day_diff"] > 300) & (df["day_diff"] <= 600), "overall"].mean() * 25 / 100)
print(df.loc[df["day_diff"] > 600, "overall"].mean() * 25 / 100)

# When equally weighted, recent reviews still tend to have higher ratings.

# Task 2: Identify the top 20 reviews to display on the product details page.

# Step 1: Create the 'helpful_no' variable
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Step 2: Calculate score_pos_neg_diff, score_average_rating, and wilson_lower_bound

def pos_neg_diff(dataframe):
    return dataframe["helpful_yes"] - dataframe["helpful_no"]

df["score_pos_neg_diff"] = pos_neg_diff(df)

def average_rating(dataframe):
    return dataframe.apply(lambda x: 0 if x["total_vote"] == 0 else x["helpful_yes"] / x["total_vote"], axis=1)

df["score_average_rating"] = average_rating(df)

def wlb(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wlb(x["helpful_yes"], x["helpful_no"]), axis=1)

# Step 3: Select the top 20 reviews based on wilson_lower_bound score
top_reviews = df.sort_values("wilson_lower_bound", ascending=False).head(20)
print(top_reviews)

# The Wilson Lower Bound score shows that not only the average rating matters,
# but also the difference between positive and negative votes has a significant impact.
