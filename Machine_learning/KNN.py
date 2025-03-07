import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set pandas options for better readability
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the dataset
df = pd.read_csv("datasets/diabetes.csv")

# Replace zero values with NaN in specific columns where zero is not a valid value
columns_with_no_zero = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness"]
for col in columns_with_no_zero:
    df[col] = df[col].replace(0, np.nan)

# Function to identify and display missing values
def missing_vals(dataframe, return_cols=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    
    if return_cols:
        return na_cols

# Identify missing columns and fill them using mean values grouped by the Outcome column
na_cols = missing_vals(df, True)
for col in na_cols:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("mean"))

# Define features and target variable
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Scale the feature variables
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train the K-Nearest Neighbors model
knn_model = KNeighborsClassifier().fit(X, y)

# Sample a random user for prediction
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user)

# Model evaluation

# Predictions for confusion matrix
y_pred = knn_model.predict(X)

# Probabilities for AUC calculation
y_prob = knn_model.predict_proba(X)[:, 1]

# Print classification report
print(classification_report(y, y_pred))

# Compute AUC score
roc_auc_score(y, y_prob)

# Cross-validation evaluation
cv_result = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_result["test_accuracy"].mean()

# Potential improvements:
# - Increase sample size
# - Improve data preprocessing
# - Feature engineering
# - Optimize model hyperparameters

# Hyperparameter tuning using GridSearchCV
knn_params = {"n_neighbors": range(2, 50)}
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
knn_gs_best.best_params_

# Train final model with optimized parameters
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

# Evaluate the final model
cv_result = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_result["test_accuracy"].mean()
