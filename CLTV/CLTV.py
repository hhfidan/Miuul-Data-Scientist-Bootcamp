from unittest.mock import inplace

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set pandas display options for better readability
pd.set_option("display.max_columns", None)  # Show all columns
# pd.set_option("display.max_rows", None)  # Uncomment to show all rows
pd.set_option("display.float_format", lambda x: "%5f" % x)  # Format floats with 5 decimal places

# Load dataset from Excel
df_ = pd.read_excel(r"C:\Users\hhfid\Desktop\miuul\crmAnalytics\datasets\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()  # Create a copy of the dataframe to work with

# Data cleaning: Remove invoices with 'C' and rows with negative or zero quantity
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)  # Drop rows with missing values

# Generate basic statistics
df.describe().T

# Calculate the total price for each row
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Aggregate data by customer ID to calculate transaction metrics
cltv_c = df.groupby("Customer ID").agg({
    "Invoice": lambda x: x.nunique(),  # Number of unique invoices per customer
    "Quantity": lambda x: x.sum(),     # Total quantity purchased by the customer
    "TotalPrice": lambda x: x.sum()    # Total spending by the customer
})

# Rename the columns for clarity
cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

# Convert customer IDs to integer type
cltv_c.index = cltv_c.index.astype(int)

# Calculate additional features
cltv_c["Avarage_Order_Value"] = cltv_c["total_price"] / cltv_c["total_transaction"]  # Average order value
cltv_c["Purchase_Frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]  # Purchase frequency (transactions per customer)
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]  # Repeat rate
Churn_Rate = 1 - repeat_rate  # Churn rate (1 - repeat rate)
cltv_c.drop("Repeat_Rate", axis=1, inplace=True)  # Drop repeat rate column

# Calculate profit margin and customer value
cltv_c["Profit_Margin"] = cltv_c["total_price"] * 0.10  # Assuming 10% profit margin
cltv_c["Customer_Value"] = cltv_c["Avarage_Order_Value"] * cltv_c["Purchase_Frequency"]  # Customer value

# Calculate CLTV (Customer Lifetime Value)
cltv_c["cltv"] = (cltv_c["Customer_Value"] / Churn_Rate) * cltv_c["Profit_Margin"]

# Sort the CLTV values in descending order and display the top 5
cltv_c.sort_values(by="cltv", ascending=False).head()

# Segment customers based on CLTV using quantiles (A, B, C, D segments)
cltv_c["Segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

# Aggregate data by segment to analyze customer groups
cltv_c.groupby("Segment").agg({"count", "mean", "sum"})

# Save the final CLTV data to a CSV file
cltv_c.to_csv("cltv_c.csv")

# Function to create the CLTV for customers in a given dataset
def create_cltv_c(dataframe, profit=0.10):
    """
    This function calculates the Customer Lifetime Value (CLTV) for each customer in the given dataframe.
    It includes steps like data cleaning, feature engineering, and CLTV calculation.
    """
    # Clean data by removing invoices with 'C' and rows with negative or zero quantity
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)

    # Calculate total price for each transaction
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # Aggregate by customer ID
    cltv_c = dataframe.groupby("Customer ID").agg({
        "Invoice": lambda x: x.nunique(),
        "Quantity": lambda x: x.sum(),
        "TotalPrice": lambda x: x.sum()
    })

    # Rename columns for clarity
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # Convert customer IDs to integers
    cltv_c.index = cltv_c.index.astype(int)

    # Calculate average order value and purchase frequency
    cltv_c["Avarage_Order_Value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    cltv_c["Purchase_Frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    
    # Calculate repeat rate and churn rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    Churn_Rate = 1 - repeat_rate
    
    # Calculate profit margin and customer value
    cltv_c["Profit_Margin"] = cltv_c["total_price"] * profit
    cltv_c["Customer_Value"] = cltv_c["Avarage_Order_Value"] * cltv_c["Purchase_Frequency"]
    
    # Calculate the final CLTV
    cltv_c["cltv"] = (cltv_c["Customer_Value"] / Churn_Rate) * cltv_c["Profit_Margin"]

    # Sort the data by CLTV in descending order and display top records
    cltv_c.sort_values(by="cltv", ascending=False).head()

    # Segment customers based on CLTV
    cltv_c["Segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    # Aggregate by segment for analysis
    cltv_c.groupby("Segment").agg({"count", "mean", "sum"})

    # Save the result to a CSV file
    cltv_c.to_csv("cltv_c.csv")

    return cltv_c

# Create CLTV for the dataframe and return the result
clv = create_cltv_c(df)
