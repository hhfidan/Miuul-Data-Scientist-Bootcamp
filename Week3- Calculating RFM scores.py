# Imports
import pandas as pd
import datetime as dt
from unittest.mock import inplace  # This import seems unused, it can be removed

# Data display settings
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  # Limiting the number of rows can be useful for large datasets
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Loading the data
file_path = "C:/Users/username/Desktop/miuul/crmAnalytics/datasets/online_retail_II.xlsx"  # Use a general path
df_ = pd.read_excel(file_path, sheet_name="Year 2009-2010")
df = df_.copy()

# Adding a new column for total price
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Dropping rows with missing values
df.dropna(inplace=True)

# Removing invoices containing 'C'
df = df[~df["Invoice"].str.contains("C", na=False)]

# General statistical overview
df.describe().T

# Defining today's date
today_date = dt.datetime(2010, 12, 11)

# Calculating RFM metrics
rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda recency: (today_date - recency.max()).days,  # Recency
    "Invoice": lambda frequency: frequency.nunique(),  # Frequency
    "TotalPrice": lambda monetary: monetary.sum()  # Monetary
})

# Displaying the first few rows
rfm.head()

# Renaming columns for clarity
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Keeping only positive monetary values
rfm = rfm[rfm["Monetary"] > 0]

# Calculating RFM scores
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

# Displaying the first few rows again
rfm.head()

# Combining the RFM scores into a single score
rfm["RFM_Score"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

# Reordering the columns
cols = rfm.columns.tolist()
cols[4], cols[5] = cols[5], cols[4]  # Swapping 'recency_score' and 'frequency_score' columns
rfm = rfm[cols]

# Final output
print(rfm.head())
