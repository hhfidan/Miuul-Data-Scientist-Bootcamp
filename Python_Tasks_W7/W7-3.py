import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from click import style

# Set pandas display options
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Load the dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week7\Telco-Customer-Churn.csv")

#############################################################################################
################################ Task 1: Exploratory Data Analysis ##########################
#############################################################################################

def checker(dataframe):
    # Print basic structure of the data
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head Tail ##############")
    print(dataframe.head())
    print(dataframe.tail())
    print("################## NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles ##############")
    print(dataframe.describe([0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]).T)

checker(df)
# 'TotalCharges' should be float, 'SeniorCitizen' should be a categorical variable (string)

# Step 1: Identify numeric and categorical variables
def grab(dataframe):
    categoric = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    numericcol = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    cardinal = [col for col in categoric if dataframe[col].nunique() > 10]
    catbutnum = [col for col in numericcol if dataframe[col].nunique() <10]
    categoric = categoric + catbutnum
    categoric = [col for col in categoric if col not in cardinal]
    numericcol = [col for col in numericcol if col not in catbutnum]
    na_values_str = [col for col in dataframe.columns if dataframe[col].dtype == "O" and (dataframe[col].str.strip()== "").any()]
    num_with_zero = [col for col in numericcol if (dataframe[col]==0).any()]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categoric Columns: {len(categoric)}")
    print(f"Numeric Columns: {len(numericcol)}")
    print(f"Categorics with High Cardinality: {len(cardinal)}")
    print(f"Categorics from Numerics: {len(catbutnum)}")
    print(f"NA Values STR: {len(na_values_str)}")
    print(f"num_with_zero: {len(num_with_zero)}")
    print("Categoric - Numeric - CardinalCats - na_values_str - num_with_zero")

    return categoric, numericcol, cardinal, na_values_str, num_with_zero

categoric, numericcol, cardinal, na_values_str, na_values_num = grab(df)

# Step 2: Handle data issues (e.g., correcting data types)
for col in na_values_str:
    df[col] = df[col].replace(" ", np.nan)

df.isnull().sum()

df["TotalCharges"] = df["TotalCharges"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
checker(df)
categoric, numericcol, cardinal, na_values_str, na_values_num = grab(df)

# Step 3: Observe the distribution of numeric and categorical variables
def cat_num_summary(dataframe, col_cat, col_num, plot=False):
    for col in col_cat:
        print(pd.DataFrame({col: dataframe[col].value_counts(),
                             "Ratio:": 100*dataframe[col].value_counts()/len(dataframe)}))
        print("###########################################")
        if plot:
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.show(block=True)

    for col in col_num:
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90, 0.95, 1]
        print(dataframe[col].describe(quantiles).T)
        if plot:
            dataframe[col].hist()
            plt.xlabel(col)
            plt.title(col)
            plt.show(block=True)

cat_num_summary(df, categoric, numericcol)

# Step 4: Analyze categorical variables with respect to the target variable (Churn)
for col in categoric:
    print(f"### {col} ###")
    print(df.groupby("Churn")[col].value_counts(normalize=True))

# Cross-tabulation between categorical variables and Churn
for col in categoric:
    print(f"### {col} Crosstab ###")
    print(pd.crosstab(df[col], df["Churn"], normalize="index"))

# Step 5: Check for outliers
def outlierFinder(dataframe, col_name, q1=0.01, q3= 0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def checkOutlier(dataframe, col_name):
    low_limit, up_limit = outlierFinder(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in numericcol:
    print(col, checkOutlier(df, col))

# Step 6: Handle missing data
df.isnull().sum()
df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("Churn")["TotalCharges"].transform("mean"))
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

#############################################################################################
############################# Task 2: Feature Engineering ##################################
#############################################################################################

# Step 1: Handle missing and outlier observations
th = np.percentile(df_scores, 5)  # Remove outliers based on LOF scores
df = df[df_scores > th]

# Step 2: Create new features
df.columns = df.columns.str.lower()
df["tenurecategory"] = pd.cut(df["tenure"], bins=[-1,12,24,36,48,100], labels=["0-1 Year", "1-2 Years", "2-3 Years", "3-4 Years", "4+ Years"])

df["monthlycharges"].describe()
df["mothlychargesvalue"] = pd.cut(df["monthlycharges"], bins=[0,40,80,120], labels=["low", "mid", "high"])

service_cols = ["phoneservice","onlinesecurity", "onlinebackup", "deviceprotection", "techsupport", "streamingtv", "streamingmovies", "internetservice"]
df["totalservice"] = df[service_cols].apply(lambda x: (~x.isin(["No","No internet service"])).sum(), axis=1)

df["autopayment"] = df["paymentmethod"].apply(lambda x: 1 if "automatic" in x.lower() else 0)
df["isfamily"] = df.apply(lambda x: 1 if x["partner"] == "Yes" or x["dependents"] == "Yes" else 0, axis=1)
df["estimatedmoths"] = df["totalcharges"] / df["monthlycharges"]
df["usageintensity"] = df["monthlycharges"] * df["tenure"]
df["chargepermonth"] = df["totalcharges"] / (df["tenure"]+1)
df["firstmonthcharge"] = df["totalcharges"] / df["tenure"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df["firstmonthcharge"].fillna(df["monthlycharges"], inplace=True)
df["chargediff"] = df["monthlycharges"] - df["firstmonthcharge"]
df["servicepercharge"] = df["totalcharges"] / (df["totalservice"] + 1)

# Encode categorical features
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Step 3: Perform encoding
df2 = df.copy()
df2 = pd.get_dummies(df2, columns=categoric, drop_first=True, dtype=np.uint8)

# Step 4: Standardize numerical variables
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
df2[numericcol] = scaler.fit_transform(df2[numericcol])

# Logistic Regression Accuracy: 0.8057
# Random Forest Accuracy: 0.7867
# Support Vector Machine Accuracy: 0.7902
# K-Nearest Neighbors Accuracy: 0.7598
# XGBoost Accuracy: 0.7838
# AdaBoost Accuracy: 0.8022

df3 = df2.copy()
scaler2 = StandardScaler()
df3[numericcol] = scaler2.fit_transform(df3[numericcol])

# Logistic Regression Accuracy: 0.8057
# Random Forest Accuracy: 0.7882
# Support Vector Machine Accuracy: 0.7902
# K-Nearest Neighbors Accuracy: 0.7583
# XGBoost Accuracy: 0.7838
# AdaBoost Accuracy: 0.8013

#############################################################################################
################################# Task 3: Modeling ########################################
#############################################################################################

# Step 1: Train models with classification algorithms and examine their accuracy scores. 
# Select the top 4 models.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

X = df3.drop(["churn_1", "customerid"], axis=1)
y = df3["churn_1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB()  # Added Naive Bayes model
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

"""
TOP 4 MODELS
Logistic Regression Accuracy: 0.8107
Random Forest Accuracy: 0.7818
Support Vector Machine Accuracy: 0.7967
AdaBoost Accuracy: 0.8012
"""

# Step 2: Perform hyperparameter optimization for the selected models and retrain with the best hyperparameters.

from sklearn.model_selection import GridSearchCV

logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
logreg = LogisticRegression()
logreg_grid = GridSearchCV(estimator=logreg, param_grid=logreg_param_grid, cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)
print(f"Best Logistic Regression Parameters: {logreg_grid.best_params_}")
# Best Logistic Regression Parameters: {'C': 1, 'solver': 'saga'}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier()
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
print(f"Best Random Forest Parameters: {rf_grid.best_params_}")
# Best Random Forest Parameters: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC()
svm_grid = GridSearchCV(estimator=svm, param_grid=svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)
print(f"Best SVM Parameters: {svm_grid.best_params_}")
# Best SVM Parameters: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}

adaboost_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1]
}
adaboost = AdaBoostClassifier()
adaboost_grid = GridSearchCV(estimator=adaboost, param_grid=adaboost_param_grid, cv=5, scoring='accuracy')
adaboost_grid.fit(X_train, y_train)
print(f"Best AdaBoost Parameters: {adaboost_grid.best_params_}")
# Best AdaBoost Parameters: {'learning_rate': 0.1, 'n_estimators': 150}

# Retraining best model with optimal parameters
best_model = AdaBoostClassifier(**adaboost_grid.best_params_)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

from sklearn.model_selection import cross_val_score
cross_val_score(best_model, X, y, cv=5).mean()

# cv = 0.8034
