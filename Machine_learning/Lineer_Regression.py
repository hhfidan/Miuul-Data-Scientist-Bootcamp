import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Homeworks.Week6 import x_train, y_test

# Setting display options for Pandas
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Importing machine learning libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Loading the dataset
df = pd.read_csv("datasets/advertising.csv")

# Defining the feature and target variable
X = df[["TV"]]
y = df[["sales"]]

### MODEL BUILDING

# Fitting a linear regression model
reg_model = LinearRegression().fit(X, y)

# Intercept (B0) and Coefficient (B1)
reg_model.intercept_[0]  # B0
reg_model.coef_[0][0]  # B1

# Prediction for a specific value (e.g., 150 units of TV spending)
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# Visualization
sns_plot = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9}, ci=False, color="r")
sns_plot.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
sns_plot.set_ylabel("Sales")
sns_plot.set_xlabel("TV Spending")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

### MODEL PERFORMANCE

# Predictions on training data
y_pred = reg_model.predict(X)

# Mean Squared Error (MSE)
mean_squared_error(y, y_pred)  # 10.51

# Root Mean Squared Error (RMSE)
np.sqrt(mean_squared_error(y, y_pred))  # 3.24

# Mean Absolute Error (MAE)
mean_absolute_error(y, y_pred)  # 2.54

# R-squared
reg_model.score(X, y)  # 0.61 (indicates that 61% of the variability in sales is explained by TV spending)

### MULTIPLE LINEAR REGRESSION

# Defining features and target variable
X = df.drop("sales", axis=1)
y = df[["sales"]]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Fitting the model
reg_model = LinearRegression().fit(X_train, y_train)

# Making a sample prediction
sample_data = [[30], [10], [40]]
sample_data = pd.DataFrame(sample_data).T
reg_model.predict(sample_data)

### MODEL PERFORMANCE

# Train RMSE
y_pred_train = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred_train))  # 1.73

# Train R-squared
reg_model.score(X_train, y_train)

# Test RMSE
y_pred_test = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))  # 1.71

# Test R-squared
reg_model.score(X_test, y_test)

# Cross-validation with 5-fold CV
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error")))  # 1.71
