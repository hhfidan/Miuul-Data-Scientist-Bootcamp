import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week6\diabetes\diabetes.csv")

# Step 1: Identifying invalid zero values in certain columns
cant_zero = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness"]
for col in cant_zero:
    df[col] = df[col].replace(0, np.nan)

# Step 2: Missing value analysis
def missing_values(dataframe):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    return na_cols

na_cols = missing_values(df)

def missing_targets(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    return temp_df

df = missing_targets(df, "Outcome", na_cols)

# Fill missing values using group means
df[na_cols] = df.groupby("Outcome")[na_cols].transform("mean")

# Fill missing values using KNN Imputation
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

# Step 3: Outlier detection and removal
clf = LocalOutlierFactor(n_neighbors=20)
df_scores = clf.fit_predict(df)
th = np.sort(df_scores)[15]
df = df[df_scores > th]

# Step 4: Feature Engineering

df["Score_with_Preg"] = (df["Pregnancies"] * df["Glucose"] * df["BloodPressure"] * df["SkinThickness"] *
                           df["Insulin"] * df["BMI"] * df["DiabetesPedigreeFunction"] * df["Age"]) / 10**9

df["Score_without_Preg"] = (df["Glucose"] * df["BloodPressure"] * df["SkinThickness"] *
                              df["Insulin"] * df["BMI"] * df["DiabetesPedigreeFunction"] * df["Age"]) / 10**9

# Age categorization
df.loc[df["Age"] < 30, "new_age_cat"] = "youngFemale"
df.loc[(df["Age"] >= 30) & (df["Age"] <= 40), "new_age_cat"] = "matureFemale"
df.loc[(df["Age"] > 40) & (df["Age"] <= 50), "new_age_cat"] = "seniorFemale"
df.loc[df["Age"] > 50, "new_age_cat"] = "oldFemale"

# Step 5: Encoding categorical features
df = pd.get_dummies(df, columns=["new_age_cat"], drop_first=True, dtype=np.uint8)

# Step 6: Standardizing numerical features
categoric_cols = [col for col in df.columns if df[col].dtypes == "O"]
numeric_cols = [col for col in df.columns if df[col].dtypes != "O"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 7: Model training and evaluation
y = df["Outcome"]
X = df.drop(["Outcome", "Pregnancies"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Feature Importance Plot
def plot_importance(model, features, num=None, save=False):
    if num is None:
        num = len(features.columns)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:num])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importance.png")

plot_importance(rf_model, X_train)
