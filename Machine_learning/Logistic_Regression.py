import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

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

# Scale the features using RobustScaler
for col in df.columns:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

# Detect and remove outliers using Local Outlier Factor
clf = LocalOutlierFactor(n_neighbors=10)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

df_scores_sorted = np.sort(df_scores)
scores_df = pd.DataFrame(df_scores_sorted)
scores_df.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

# Set threshold and filter out outliers
threshold = df_scores_sorted[2]
df = df[df_scores > threshold]

# Define features and target variable
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Train Logistic Regression model
log_model = LogisticRegression().fit(X, y)

# Model coefficients
log_model.intercept_
log_model.coef_

# Predictions
y_pred = log_model.predict(X)

# Function to plot confusion matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Accuracy Score: {acc}", size=10)
    plt.show()

# Evaluate the model
plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))

# Compute ROC AUC score
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

# Train-Test Split for model validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Plot ROC Curve
RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()

roc_auc_score(y_test, y_prob)

# 5-Fold Cross Validation
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y)

cv_result = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# Example prediction on a random sample
random_user = X.sample(1)
log_model.predict(random_user)
