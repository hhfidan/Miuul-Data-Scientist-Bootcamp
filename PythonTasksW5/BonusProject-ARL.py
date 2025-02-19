# Association Rule Based Recommender System

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.signal import freqs
from streamlit import dataframe
from sympy.physics.units import frequency

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

#### Görev 1: Veriyi Hazırlama #####

df_ = pd.read_excel(r"C:\Users\hhfid\Desktop\miuul\week5\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df=df.dropna()

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

def outlier_th(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile
    low_limit = quartile1 - 1.5 * interquantile
    return low_limit, up_limit

def replace_with_th(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_th(df, "Quantity")
replace_with_th(df, "Quantity")
outlier_th(df, "Price")
replace_with_th(df, "Price")

df = df[~df["StockCode"].str.contains("POST", na=False)]

##### Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme ############

# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.

def create_invoice_product_df(dataframe, id=True):
    if id:

         return dataframe.pivot_table(index="Invoice", columns="StockCode", values="Quantity").fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:

        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

create_invoice_product_df(df)

# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.

def create_rules(dataframe, country = "Germany", id=True):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    dataframe = dataframe.astype(bool)
    frequent_items = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="support", min_threshold=0.01)
    return rules

a = create_rules(df)


######## Görev 2: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma ####

# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, ID):

    product_name = dataframe.loc[dataframe["StockCode"] == ID, "Description"].values[0]
    print(product_name)

# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

def arl_recommender(dataframe, ID, rec_count=1):
    sorted_rules = dataframe.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == ID:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

arl_recommender(a, 22747,2)
arl_recommender(a, 21987,2)
arl_recommender(a, 23235,2)

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

check_id(df, arl_recommender(a, 23235,2)[0])
