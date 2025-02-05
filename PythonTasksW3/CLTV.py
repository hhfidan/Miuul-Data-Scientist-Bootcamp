import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

# Set display options for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%4f" % x)

# Step 1: Read the dataset

df = pd.read_csv("flo_data_20k.csv")
df = df.copy()

# Step 2: Define functions to handle outliers

def outliers_threshold(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return round(low_limit), round(up_limit)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# Step 3: Handle outliers for relevant columns

columns_to_check = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online"
]

for col in columns_to_check:
    replace_with_thresholds(df, col)

# Step 4: Convert date columns to datetime format

df[[col for col in df.columns if "date" in col.lower()]] = df[[col for col in df.columns if "date" in col.lower()]].apply(pd.to_datetime)

# Step 5: Define the analysis date

today_date = dt.datetime(2021, 6, 1)

# Step 6: Create CLTV dataframe

cltv_p = df.groupby("master_id").agg(
    first_order_date=("first_order_date", "min"),
    last_order_date=("last_order_date", "max"),
    T=("first_order_date", lambda x: (today_date - x.min()).days),
    frequency=("Order_num_total_omnichannel", "sum"),
    monetary=("total_price", "sum")
)

cltv_p["recency"] = (cltv_p["last_order_date"] - cltv_p["first_order_date"]).dt.days
cltv_p = cltv_p.drop(["first_order_date", "last_order_date"], axis=1)

# Convert values to weekly metrics
cltv_p["recency"] /= 7
cltv_p["T"] /= 7
cltv_p["monetary"] = cltv_p["monetary"] / cltv_p["frequency"]

# Step 7: Fit the BG/NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_p["frequency"], cltv_p["recency"], cltv_p["T"])

# Predict expected sales for 3 and 6 months
cltv_p["exp_sales_3_month"] = bgf.predict(4 * 3, cltv_p["frequency"], cltv_p["recency"], cltv_p["T"])
cltv_p["exp_sales_6_month"] = bgf.predict(4 * 6, cltv_p["frequency"], cltv_p["recency"], cltv_p["T"])

# Plot transaction history
plot_period_transactions(bgf)
plt.show()

# Step 8: Fit the Gamma-Gamma model

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_p["frequency"], cltv_p["monetary"])

# Predict expected average monetary value
cltv_p["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_p["frequency"], cltv_p["monetary"])

# Step 9: Calculate 6-month CLTV

cltv = ggf.customer_lifetime_value(
    bgf,
    cltv_p["frequency"],
    cltv_p["recency"],
    cltv_p["T"],
    cltv_p["monetary"],
    time=6,
    freq="W",
    discount_rate=0.01
)

cltv_p["cltv"] = cltv
cltv_p = cltv_p.reset_index()

# Display top 20 customers by CLTV
print(cltv_p.sort_values(by="cltv", ascending=False).head(20))

# Step 10: Create customer segments

cltv_p["segment"] = pd.qcut(cltv_p["cltv"], 4, labels=["D", "C", "B", "A"])

# Summarize segment statistics
print(cltv_p.groupby("segment").agg({"cltv": ["count", "mean", "sum"]}).reset_index())
