"""
master_id: Unique customer ID  
order_channel: The platform channel through which the purchase was made (Android, iOS, Desktop, Mobile)  
last_order_channel: The channel used for the most recent purchase  
first_order_date: The date of the customer's first purchase  
last_order_date: The date of the customer's most recent purchase  
last_order_date_online: The date of the customer's most recent purchase on an online platform  
last_order_date_offline: The date of the customer's most recent purchase on an offline platform  
order_num_total_ever_online: The total number of purchases the customer has made on online platforms  
order_num_total_ever_offline: The total number of purchases the customer has made on offline platforms  
customer_value_total_ever_offline: The total amount spent by the customer on offline purchases  
customer_value_total_ever_online: The total amount spent by the customer on online purchases  
interested_in_categories_12: The list of categories the customer has shopped from in the last 12 months  
"""

import pandas as pd
import datetime as dt

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%4f" % x)

# Step 1: Read the flo_data_20K.csv dataset and create a copy of the dataframe.
df_ = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\flo_data_20k.csv")
df = df_.copy()

# Step 2: Explore the dataset
df.head(10)  # First 10 observations
df.columns  # Variable names
df.describe().T  # Descriptive statistics
df.isnull().sum()  # Missing values
df.info()  # Variable types

# Step 3: Create new variables for total purchases and total spending per customer.
df["Order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["purchase_freq"] = df["Order_num_total_omnichannel"] / df.shape[0]
df["average_order_value"] = df["customer_value_total_ever_omnichannel"] / df["purchase_freq"]
df["total_price"] = df["average_order_value"] / df["Order_num_total_omnichannel"]

# Step 4: Convert date-related columns to datetime format.
date_cols = [col for col in df.columns if "date" in col.lower()]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

# Step 5: Analyze customer distribution by order channel.
df.groupby("order_channel").agg({
    "master_id": "count",
    "Order_num_total_omnichannel": ["sum", "mean"],
    "total_price": ["sum", "mean"]
})

# Step 6: Identify the top 10 customers by total revenue.
df.sort_values(by="total_price", ascending=False)[["master_id", "total_price"]].head(10)

# Step 7: Identify the top 10 customers by total orders.
df.sort_values(by="Order_num_total_omnichannel", ascending=False)[["master_id", "Order_num_total_omnichannel"]].head(10)

# Step 8: Define a function for data preprocessing.
def prepare_data(dataframe):
    dataframe["Order_num_total_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever_omnichannel"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe["purchase_freq"] = dataframe["Order_num_total_omnichannel"] / dataframe.shape[0]
    dataframe["average_order_value"] = dataframe["customer_value_total_ever_omnichannel"] / dataframe["purchase_freq"]
    dataframe["total_price"] = dataframe["average_order_value"] / dataframe["Order_num_total_omnichannel"]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)
    return dataframe

prepare_data(df)

# Task 2: Calculate RFM Metrics
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({
    "last_order_date": lambda x: (today_date - x.max()).days,
    "Order_num_total_omnichannel": "sum",
    "total_price": "sum"
})

# Rename columns to recency, frequency, and monetary.
rfm.columns = ["recency", "frequency", "monetary"]

# Task 3: Calculate RF Score
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["freq_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_score"] = rfm["recency_score"].astype(str) + rfm["freq_score"].astype(str)

# Task 4: Define RF Score Segments
seg_map = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At_Risk",
    r"[1-2]5": "Cant_Lose",
    r"3[1-2]": "About_to_Sleep",
    r"33": "Need_Attention",
    r"[3-4][4-5]": "Loyal_Customers",
    r"41": "Promising",
    r"51": "New_Customers",
    r"[4-5][2-3]": "Potential_Loyalists",
    r"5[4-5]": "Champions"
}
rfm["segment"] = rfm["RF_score"].replace(seg_map, regex=True)

# Task 5: Action Time!
rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()

# Targeting specific customer segments
# A - Champions, Loyal_Customers, and Female category
filtered_df = df[df["master_id"].isin(rfm[rfm["segment"].isin(["Champions", "Loyal_Customers"])].index)]
target_customers = filtered_df[filtered_df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
target_customers.to_csv("target_customers.csv", index=False)

# B - Cant_Lose, About_to_Sleep, New_Customers, Male or Children category
filtered_df2 = df[df["master_id"].isin(rfm[rfm["segment"].isin(["Cant_Lose", "About_to_Sleep", "New_Customers"])].index)]
target_customers2 = filtered_df2[filtered_df2["interested_in_categories_12"].str.contains("ERKEK|COCUK")]["master_id"]
target_customers2.to_csv("target_customers2.csv", index=False)
