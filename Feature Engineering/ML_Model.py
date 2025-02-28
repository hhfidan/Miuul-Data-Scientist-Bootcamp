import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Display all columns and adjust display width
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the Titanic dataset
df = pd.read_csv(r"C:\Users\hhfid\Desktop\miuul\feature_engineering\datasets\titanic.csv")

# Drop missing values before proceeding
# Ensure encoding steps are done before running the model
df = df.dropna()

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Define target variable (y) and features (X)
y = df["Survived"]
x = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

# Train the Random Forest Classifier model
rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(x_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_pred, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Function to plot feature importance
def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    
    if save:
        plt.savefig("importance.png")

# Plot the feature importance
plot_importance(rf_model, x_train)
