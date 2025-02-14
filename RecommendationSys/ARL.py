import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules

# Configure pandas display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

# Load dataset
file_path = r"C:\Users\hhfid\Desktop\miuul\recommender_systems\datasets\online_retail_II.xlsx"
df_ = pd.read_excel(file_path, sheet_name="Year 2010-2011")
df = df_.copy()

# Convert date column to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
# Convert customer ID to object type
df["Customer ID"] = df["Customer ID"].astype("object")

# Function to calculate outlier thresholds
def outlier_th(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Function to replace outliers with threshold values
def replace_with_th(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Function to clean and preprocess the dataset
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # Remove canceled invoices
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_th(dataframe, "Quantity")
    replace_with_th(dataframe, "Price")
    return dataframe

# Apply preprocessing
df = retail_data_prep(df)

# Filter transactions for a specific country (France)
df_fr = df[df["Country"] == "France"]

# Function to create an invoice-product matrix
def create_inv_prod_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Create invoice-product matrix for France
fr_inv_pro_df = create_inv_prod_df(df_fr, id=True)

# Function to check product name by stock code
def check_id(dataframe, stock_code):
    product_name = dataframe.loc[dataframe["StockCode"] == stock_code, "Description"].values
    print(product_name)

# Example usage
check_id(df, 11001)

# Convert dataframe to boolean format
fr_inv_pro_df = fr_inv_pro_df.astype(bool)

# Generate frequent item sets using the Apriori algorithm
frequent_items = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_items, metric="support", min_threshold=0.01)

# Filter and sort association rules based on confidence and lift
filtered_rules = rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]
filtered_rules.sort_values("confidence", ascending=False, inplace=True)

# Explanation of the output variables:
# antecedents: First product in the rule
# consequents: Second product in the rule
# antecedent support: Probability of observing the first product alone
# consequent support: Probability of observing the second product alone
# support: Probability of both products being observed together
# confidence: Probability of buying the second product given that the first one is bought
# lift: How many times more likely the second product is bought when the first product is purchased
# leverage: Difference between observed support and expected support under independence
# conviction: Expected frequency of the second product when the first is not bought

# Function to automate rule creation for a given country
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_inv_prod_df(dataframe, id)
    dataframe = dataframe.astype(bool)
    frequent_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="support", min_threshold=0.01)
    return rules

# Apply rule creation function
rules = create_rules(df)
filtered_rules = rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]
filtered_rules.sort_values("confidence", ascending=False, inplace=True)
