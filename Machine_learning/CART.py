import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore", category=Warning)

# Load dataset
df = pd.read_csv("datasets/diabetes.csv")

# Target and Features
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Fit the CART model
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Predictions for confusion matrix
y_pred = cart_model.predict(X)

# Predictions for AUC score
y_prob = cart_model.predict_proba(X)[:, 1]

# Classification report and AUC score
print(classification_report(y, y_pred))
print("AUC Score:", roc_auc_score(y, y_prob))

# Holdout method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train error
y_pred_train = cart_model.predict(X_train)
y_prob_train = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred_train))
print("Train AUC:", roc_auc_score(y_train, y_prob_train))

# Cross-validation performance evaluation
cv_result = cross_validate(cart_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print("CV Accuracy:", cv_result["test_accuracy"].mean())
print("CV F1 Score:", cv_result["test_f1"].mean())
print("CV AUC Score:", cv_result["test_roc_auc"].mean())

# Hyperparameter tuning
cart_params = {
    "max_depth": range(1, 11),
    "min_samples_split": range(2, 20)
}

cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Best parameters and best score
print("Best Parameters:", cart_best_grid.best_params_)
print("Best Score:", cart_best_grid.best_score_)

# Final Model
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)

# Feature Importance Plot
def plot_importance(model, features, num=None, save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:num])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

plot_importance(cart_final, X)

# Validation Curve
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

val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

# Decision Tree Visualization
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

# Decision Rules
print(export_text(cart_final, feature_names=list(X.columns)))

# Save and Load Model
joblib.dump(cart_final, "cart_final.plk")
cart_model_from_disk = joblib.load("cart_final.plk")

# Sample Prediction
sample = df.sample(1).drop("Outcome", axis=1)
prediction = cart_model_from_disk.predict(sample)
print("Sample Prediction:", prediction)
