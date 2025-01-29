import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Display settings for Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load Titanic dataset
df = sns.load_dataset("titanic")

def grab_col_names(dataframe, cat_th=10, car_th=10):
    """
    Identifies categorical, numerical, and categorical but cardinal variables in the given dataset.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe from which variable names will be extracted.
    cat_th : int, float, default=10
        Threshold value for numerical but categorical variables.
    car_th : int, float, default=10
        Threshold value for categorical but cardinal variables.

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numerical variables.
    cat_but_car : list
        List of categorical but cardinal variables.

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables.
    num_but_cat variables are included in cat_cols.
    """
    
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object"] and dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(f"Numerical Columns: {len(num_cols)}")
    print(f"Categorical but Cardinal: {len(cat_but_car)}")
    print(f"Numerical but Categorical: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

# Retrieve column types
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Print column types
print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)
print("Categorical but Cardinal Columns:", cat_but_car)
