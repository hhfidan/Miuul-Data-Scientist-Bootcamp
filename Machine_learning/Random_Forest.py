import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore", category=Warning)

# Load dataset
df = pd.read_csv("datasets/diabetes.csv")

# Define target variable and features
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Initialize Random Forest model
rf_model = RandomForestClassifier()

# Perform cross-validation
cv_result = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print("Initial Model Performance:")
print(f"Accuracy: {cv_result['test_accuracy'].mean():.4f}")
print(f"F1 Score: {cv_result['test_f1'].mean():.4f}")
print(f"ROC AUC: {cv_result['test_roc_auc'].mean():.4f}")

# Define hyperparameter grid for tuning
rf_params = {
    "max_depth": [5, 7, 9, 11, None],
    "max_features": [3, 5, 7, 9, 11, "sqrt"],
    "min_samples_split": [2, 5, 8, 15, 20],
    "n_estimators": [50, 100, 200, 350, 500]
}

# Perform Grid Search to find the best hyperparameters
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

print("Best Parameters:")
print(rf_best_grid.best_params_)

# Train final model with best parameters
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Perform cross-validation on final model
cv_result = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print("Final Model Performance:")
print(f"Accuracy: {cv_result['test_accuracy'].mean():.4f}")
print(f"F1 Score: {cv_result['test_f1'].mean():.4f}")
print(f"ROC AUC: {cv_result['test_roc_auc'].mean():.4f}")

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

plot_importance(rf_final, X)

# Function to plot validation curve
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv
    )
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")
