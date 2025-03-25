import numpy as np
import pandas as pd
from astropy.utils.metadata.utils import dtype

# Set Pandas display options
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

# Load datasets
df_Att = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week9\Scoutium-220805-075951\scoutium_attributes.csv", sep=";")
df_pot = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\HW\week9\Scoutium-220805-075951\scoutium_potential_labels.csv", sep=";")

# Merge datasets on common columns
df = df_pot.merge(df_Att, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="inner")

# Filter out position_id equal to 1
df = df[df["position_id"] != 1]

# Check unique values of position_id and potential_label
df["position_id"].unique()
df["potential_label"].unique()

# Filter out records where potential_label is 'below_average'
df = df[df["potential_label"] != "below_average"]

# Create a pivot table for attributes
df_table = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns=["attribute_id"], values=["attribute_value"]).reset_index()

# Flatten column names for ease of use
df_table.columns = ["_".join(map(str, col)).strip() for col in df_table.columns]

# Display first few rows of the table
df_table.head()

# Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Label encoding for the potential label
le = LabelEncoder()
df_table["potential_label_"] = le.fit_transform(df_table["potential_label_"])

# Identify numerical columns (excluding non-relevant ones)
numcols = [col for col in df_table.columns if df_table[col].dtype != "O"]
numcols = [col for col in numcols if col not in ["player_id_", "position_id_", "potential_label_"]]

# Standardize numerical features
ss = StandardScaler()
df_table[numcols] = ss.fit_transform(df_table[numcols])

# One-hot encoding for the position_id feature
df_table = pd.get_dummies(df_table, columns=["position_id_"], drop_first=True, dtype=np.uint8)

# Separate features (X) and target (y)
y = df_table["potential_label_"]
X = df_table.drop(["player_id_", "potential_label_"], axis=1)

# Import Helper functions
from Helpers import *

# Model Training and Hyperparameter Optimization
base_models(X, y)  # Base model training

# Hyperparameter optimization
best_models = hyperparameter_optimization(X, y, 5)

# Voting Classifier to combine models
voting_classifier(best_models, X, y)

# Example output for Voting Classifier performance
"""
Voting Classifier:     
Accuracy: 0.8560439560439561
F1 Score: 0.5117283950617284
ROC AUC: 0.881014781563542
"""

# Model evaluation using Random Forest Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier and train
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set and evaluate accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation results for the Random Forest model
cv_result = cross_validate(rf_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_result['test_accuracy'].mean()}")
print(f"F1 Score: {cv_result['test_f1'].mean()}")
print(f"ROC AUC: {cv_result['test_roc_auc'].mean()}")

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

# Plot feature importance for the Random Forest model
plot_importance(rf_model, X_train)
