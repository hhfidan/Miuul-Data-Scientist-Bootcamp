import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Sample true labels and predicted probabilities
y = np.array([1,1,1,1,1,1,0,0,0,0])
y_prob = np.array([0.7,0.8,0.65,0.9,0.45,0.5,0.55,0.35,0.4,0.25])
y_pred = (y_prob >= 0.5).astype(int)

# Creating a confusion matrix with a threshold of 0.5
cm = confusion_matrix(y, y_pred)
cm_reordered = cm[::-1, ::-1]
print("Confusion Matrix--> \n", cm_reordered)
#[[5 1] TP - FP
#[1 3]] FN - TN

plt.figure(figsize=(5,4))
sns.heatmap(cm_reordered, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Churn (1)", "Non-Churn (0)"], yticklabels=["Churn (1)", "Non-Churn (0)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Calculating accuracy, recall, precision, and F1-score
acc = accuracy_score(y, y_pred)
rec = recall_score(y, y_pred)
pre = precision_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"Accuracy: {acc:.2f}\n"
      f"Recall: {rec:.2f}\n"
      f"Precision: {pre:.2f}\n"
      f"F1-score: {f1:.2f}")

###############################################

# Second scenario: Unbalanced class distribution
# Fraud cases vs Non-Fraud cases
fraud_cm = np.array([[5, 5],   # True Positive - False Positive
                      [90, 900]])  # False Negative - True Negative

acc2 = (5 + 900) / (5 + 900 + 5 + 90)  # 0.905
rec2 = 5 / (5 + 90)  # 0.052
pre2 = 5 / (5 + 5)  # 0.5
f1_2 = (2 * pre2 * rec2) / (pre2 + rec2)  # 0.095

# Interpretation: Accuracy alone can be misleading in imbalanced datasets.
# In this case, precision drops significantly from 90% to 50%, showing the importance of considering multiple metrics.
