import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Ignore warnings
warnings.simplefilter(action="ignore", category=Warning)

# Set pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load dataset
df = pd.read_csv("datasets/diabetes.csv")

# Define target and features
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# XGBoost Model
xgboost_model = XGBClassifier(random_state=17)
cv_result = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Hyperparameter tuning for XGBoost
xgb_params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 7, 10],
    "n_estimators": [350, 750, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}
xgb_best_params = GridSearchCV(xgboost_model, xgb_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
xgb_final = xgboost_model.set_params(**xgb_best_params.best_params_, random_state=17).fit(X, y)

# Feature Importance Plot
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

plot_importance(xgb_final, X)

# LightGBM Model
lgbm_model = LGBMClassifier(random_state=17)
lgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 350, 750, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# CatBoost Model
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_params = {
    "iterations": [200, 500, 1000],
    "learning_rate": [0.01, 0.1],
    "depth": [3, 6, 9]
}
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
cat_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier(random_state=17)
rf_random_params = {
    "max_depth": np.random.randint(5, 50, 10),
    "max_features": [3, 5, 7, 9, "sqrt", "auto", 11],
    "min_samples_split": np.random.randint(2, 50, 20),
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1500, num=10)]
}
rf_random = RandomizedSearchCV(
    estimator=rf_model, param_distributions=rf_random_params, n_iter=100, cv=5, verbose=True, random_state=42, n_jobs=-1
)
rf_random.fit(X, y)
rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)
