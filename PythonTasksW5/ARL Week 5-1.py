import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\week5\armut_data.csv")

df.head()

# Task 1: Data Preparation
# Step 2: ServiceID represents a different service for each CategoryID.
# Create a new variable by combining ServiceID and CategoryID with "_".

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")

df["CartID"] = df["UserId"].astype(str) + "_" + df["New_Date"]

# Task 2: Generate Association Rules and Make Recommendations
# Step 1: Create a basket-service pivot table.

service_pivot = df.groupby(["CartID", "Service"])['UserId'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

df.info()

# Step 2: Generate association rules.

service_pivot = service_pivot.astype(bool)

frequent_items = apriori(service_pivot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="support", min_threshold=0.01)

filtered_rules = rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)

sorted_rules = rules.sort_values("lift", ascending=False)

# Step 3: Use the arl_recommender function to suggest services for a user who last purchased service "2_0".

def arl_recommender(dataframe, product_id, rec_count):
    sorted_rules = dataframe.sort_values("lift", ascending=False)
    recomm_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recomm_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recomm_list[:rec_count]

arl_recommender(sorted_rules, "2_0", 2)

# Explanation of output variables:
# antecedents: First product
# consequents: Second product
# antecedent support: Probability of the first product appearing alone
# consequent support: Probability of the second product appearing alone
# support: Probability of both products appearing together
# confidence: Probability of purchasing the second product given that the first product is purchased
# lift: How many times more likely the second product is to be purchased when the first product is purchased
# leverage: Measures the strength of the association
# conviction: Expected frequency of the second product appearing without the first product
