import numpy as np
import pandas as pd

# Set display options for better readability
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Create a sample dataset
data = {
    "experience": [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    "salary": [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
}

df = pd.DataFrame(data)

# 1 - Define the linear regression equation using given bias and weight
# Equation: y' = b + w * x
bias = 275  # Intercept (b)
weight = 90  # Coefficient (w)

# 2 - Predict salaries using the defined equation
df["salary_pred"] = bias + weight * df["experience"]

# 3 - Calculate error metrics: MSE, RMSE, MAE

df["error"] = df["salary"] - df["salary_pred"]  # Difference between actual and predicted salary

df["error_squared"] = df["error"] ** 2  # Squared error

df["absolute_error"] = df["error"].abs()  # Absolute error

# Mean Squared Error (MSE)
mse = sum(df["error_squared"]) / len(df)  # 4438.333

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)  # 66.62

# Mean Absolute Error (MAE)
mae = sum(df["absolute_error"]) / len(df)  # 54.33

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
