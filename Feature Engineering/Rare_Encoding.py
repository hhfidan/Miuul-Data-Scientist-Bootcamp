import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from EDA.CVandGeneralizingOP import grab_col_names
from EDA.EDA import cat_summary

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\feature_engineering\datasets\application_train.csv")

df.head()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

def rare_analyzer(dataframe, target, cat_col):
    for col in cat_col:
        print(col, ":", len(dataframe[col].value_counts()), ":", dataframe[col].dtype)
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n")

rare_analyzer(df,"TARGET", cat_cols)

################ ####################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyzer(new_df, "TARGET", cat_cols)

