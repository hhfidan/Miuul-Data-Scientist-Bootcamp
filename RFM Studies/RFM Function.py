import pandas as pd
import datetime as dt

def create_rfm(dataframe, csv=False):
    """
    Creates an RFM (Recency, Frequency, Monetary) segmentation from the given dataframe.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing transaction data.
    csv (bool, optional): If True, saves the resulting RFM table as 'rfm.csv'. Defaults to False.

    Returns:
    pd.DataFrame: A DataFrame containing RFM scores and customer segmentation.
    """

    # Data preprocessing
    dataframe["Total_Price"] = dataframe["Quantity"] * dataframe["Price"]  # Calculate total spending
    dataframe.dropna(inplace=True)  # Remove missing values
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # Exclude canceled transactions

    # Calculate RFM metrics
    today_date = dt.datetime(2011, 12, 11)  # Reference date for recency calculation
    rfm = dataframe.groupby("Customer ID").agg({
        "InvoiceDate": lambda date: (today_date - date.max()).days,  # Recency: days since last purchase
        "Invoice": lambda num: num.nunique(),  # Frequency: number of unique transactions
        "Total_Price": lambda price: price.sum()  # Monetary: total amount spent
    })

    # Rename columns
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Remove customers with zero or negative monetary values
    rfm = rfm[rfm["Monetary"] > 0]

    # Calculate RFM scores (assigning quantile-based ranks)
    rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["FreqScore"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["MoneyScore"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    # Create combined RFM score
    rfm["RFMscore"] = rfm["RecencyScore"].astype(str) + rfm["FreqScore"].astype(str)

    # Define customer segments based on RFM scores
    seg_map = {
        r"[1-2][1-2]": "Hibernating",
        r"[1-2][3-4]": "At_Risk",
        r"[1-2]5": "Cant_Loose",
        r"3[1-2]": "About_to_Sleep",
        r"33": "Need_Attention",
        r"[3-4][4-5]": "Loyal_Customers",
        r"41": "Promising",
        r"51": "New_Customers",
        r"[4-5][2-3]": "Potential_Loyalists",
        r"5[4-5]": "Champions"
    }

    # Assign segments based on RFM scores
    rfm["Segment"] = rfm["RFMscore"].replace(seg_map, regex=True)

    # Keep relevant columns
    rfm = rfm[["Recency", "Frequency", "Monetary", "Segment"]]

    # Convert index to integer for better readability
    rfm.index = rfm.index.astype(int)

    # Save to CSV if required
    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

# Create a copy of the dataset for processing
df_new = df_.copy()

# Generate RFM analysis
rfm_new = create_rfm(df_new)

# Display the first few rows of the RFM table
print(rfm_new.head())
