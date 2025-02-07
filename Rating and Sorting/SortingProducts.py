import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Pandas display settings for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Load dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\measurement_problems\datasets\movies_metadata.csv", low_memory=False)

# Fix column naming issue
df.columns = [col.replace("commment_count", "comment_count") for col in df.columns]

# Scale purchase and comment counts to a range of 1-5
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["comment_count"]]).transform(df[["comment_count"]])

# Compute weighted sorting score
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return dataframe["comment_count_scaled"] * w1 / 100 + dataframe["purchase_count_scaled"] * w2 / 100 + dataframe["rating"] * w3 / 100

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# Compute Bayesian Average Rating
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    
    first_part = sum((k+1) * (n[k] + 1) / (N+K) for k in range(K))
    second_part = sum((k+1) ** 2 * (n[k] + 1) / (N+K) for k in range(K))
    
    score = first_part - z * math.sqrt((second_part - first_part ** 2) / (N + K + 1))
    return score

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]].tolist()), axis=1)

df.sort_values("bar_score", ascending=False).head()

# Compute Hybrid Sorting Score
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]].tolist()), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w / 100 + wss_score * wss_w / 100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)

# Compute vote count scaling
df = df[["title", "vote_average", "vote_count"]].copy()
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values("average_count_score", ascending=False).head(20)

# Compute Weighted Rating
M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)
df.sort_values("weighted_rating", ascending=False).head(10)
