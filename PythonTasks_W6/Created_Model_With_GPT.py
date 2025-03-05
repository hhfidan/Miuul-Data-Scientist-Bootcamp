import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. Load the Dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week6\diabetes\diabetes.csv")

# 2. Handle Missing and Zero Values
# Some features should not have zero values (e.g., Glucose, BloodPressure, etc.)
cantZero = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness"]
for col in cantZero:
    df[col] = df[col].replace(0, np.nan)

# 3. Normalize Data and Impute Missing Values
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

# 4. Detect and Remove Outliers Using Local Outlier Factor (LOF)
clf = LocalOutlierFactor(n_neighbors=10)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
th = np.sort(df_scores)[9]  # Threshold to remove extreme outliers
df = df[df_scores > th]

# 5. Feature Engineering (Creating New Features)
df.loc[:, "Score_with_Preg"] = (df["Pregnancies"] * df["Glucose"] * df["BloodPressure"] * df["SkinThickness"] *
                                df["Insulin"] * df["BMI"] * df["DiabetesPedigreeFunction"] * df["Age"]) / 10**9
df.loc[:, "Score_without_Preg"] = (df["Glucose"] * df["BloodPressure"] * df["SkinThickness"] *
                                   df["Insulin"] * df["BMI"] * df["DiabetesPedigreeFunction"] * df["Age"]) / 10**9

df.loc[:, "BMI_Age"] = df["BMI"] * df["Age"]

# Categorizing Age Groups
df.loc[df["Age"] < 30, "new_age_cat"] = "youngFemale"
df.loc[(df["Age"] > 30) & (df["Age"] <= 40), "new_age_cat"] = "matureFemale"
df.loc[(df["Age"] > 40) & (df["Age"] <= 50), "new_age_cat"] = "seniorFemale"
df.loc[df["Age"] > 50, "new_age_cat"] = "oldFemale"

# 6. Convert Categorical Variables
df = pd.get_dummies(df, columns=["new_age_cat"], drop_first=True, dtype=np.uint8)

# 7. Feature Selection Using SelectKBest
X = df.drop(["Outcome", "new_age_cat_seniorFemale", "new_age_cat_youngFemale"], axis=1)
y = df["Outcome"]
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 8. Split Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

# 9. Balance the Data Using SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# 10. Train and Compare Models (RandomForest vs XGBoost)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42).fit(x_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(x_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(x_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# 11. Analyze Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns[selector.get_support()]
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
print(importance_df.sort_values(by="Importance", ascending=False))
