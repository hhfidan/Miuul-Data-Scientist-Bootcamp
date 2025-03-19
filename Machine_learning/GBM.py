import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

# Set pandas options for better visibility
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Ignore warnings
warnings.simplefilter(action="ignore", category=Warning)

# Load the dataset
df = pd.read_csv("datasets/diabetes.csv")

# Define target variable and feature set
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Initialize Gradient Boosting Model
gbm_model = GradientBoostingClassifier(random_state=17)

# Display model parameters
gbm_model.get_params()

# Perform cross-validation
cv_result = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Print cross-validation results
print("Accuracy:", cv_result["test_accuracy"].mean())  # 0.8914
print("F1 Score:", cv_result["test_f1"].mean())  # 0.8406
print("ROC AUC:", cv_result["test_roc_auc"].mean())  # 0.9551

# Define hyperparameter grid
gbm_params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7, 10],
    "n_estimators": [100, 500, 750, 1000],
    "subsample": [1, 0.5, 0.7]
}

# Perform Grid Search for Hyperparameter Tuning
gbm_best_params = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

# Retrieve best parameters
print("Best Parameters:", gbm_best_params.best_params_)

# Train the final model with best parameters
gbm_final = gbm_model.set_params(**gbm_best_params.best_params_, random_state=17).fit(X, y)

# Perform cross-validation on the final model
cv_result = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Print final cross-validation results
print("Final Accuracy:", cv_result["test_accuracy"].mean())  # 0.8969
print("Final F1 Score:", cv_result["test_f1"].mean())  # 0.8503
print("Final ROC AUC:", cv_result["test_roc_auc"].mean())  # 0.9594

# Function to plot feature importance
def plot_importance(model, features, num=None, save=False):
    if num is None:
        num = len(features.columns)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})

    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:num])

    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig("importance.png")

# Plot feature importance
plot_importance(gbm_final, X)
