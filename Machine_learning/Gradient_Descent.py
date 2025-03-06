import numpy as np
import pandas as pd

# Load dataset
# The dataset contains advertising data with different types of media (TV, radio, newspaper) 
# and their impact on sales. Here, we focus on "radio" as the independent variable.
df = pd.read_csv("datasets/advertising.csv")

# Cost function (Mean Squared Error - MSE)
def cost_func(Y, b, w, X):
    """
    Computes the Mean Squared Error (MSE) between predicted and actual values.
    
    Parameters:
    Y : array-like : Actual target values (sales)
    b : float : Intercept (bias term)
    w : float : Weight (coefficient for X)
    X : array-like : Feature values (radio advertising budget)
    
    Returns:
    mse : float : Mean Squared Error
    """
    m = len(Y)
    sse = 0
    
    for i in range(m):
        y_hat = b + w * X[i]  # Predicted value
        y = Y[i]  # Actual value
        sse += (y_hat - y) ** 2  # Sum of squared errors
    
    mse = sse / m  # Mean Squared Error
    return mse

# Update weights using Gradient Descent
def update_weights(Y, b, w, X, learning_rate):
    """
    Updates the model parameters (b and w) using gradient descent.
    
    Parameters:
    Y : array-like : Actual target values
    b : float : Current intercept (bias term)
    w : float : Current weight (coefficient for X)
    X : array-like : Feature values
    learning_rate : float : Step size for updating parameters
    
    Returns:
    new_b : float : Updated intercept
    new_w : float : Updated weight
    """
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    
    for i in range(m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)  # Derivative w.r.t. b
        w_deriv_sum += (y_hat - y) * X[i]  # Derivative w.r.t. w
    
    new_b = b - (learning_rate * (1 / m) * b_deriv_sum)
    new_w = w - (learning_rate * (1 / m) * w_deriv_sum)
    
    return new_b, new_w

# Training function using Gradient Descent
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """
    Trains a simple linear regression model using gradient descent.
    
    Parameters:
    Y : array-like : Target variable
    initial_b : float : Initial value of intercept
    initial_w : float : Initial value of weight
    X : array-like : Feature variable
    learning_rate : float : Learning rate for gradient descent
    num_iters : int : Number of iterations
    
    Returns:
    cost_hist : list : List of cost values at each iteration
    final_b : float : Final learned intercept
    final_w : float : Final learned weight
    """
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(
        initial_b, initial_w, cost_func(Y, initial_b, initial_w, X)))
    
    b = initial_b
    w = initial_w
    cost_hist = []
    
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_func(Y, b, w, X)
        cost_hist.append(mse)
        
        if i % 100 == 0:  # Print progress every 100 iterations
            print("iter={:d}    b={:.2f}     w={:.4f}     mse={:.4}".format(i, b, w, mse))
    
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_func(Y, b, w, X)))
    return cost_hist, b, w

# Define feature (X) and target (Y) variables
X = df["radio"]  # Independent variable
Y = df["sales"]  # Dependent variable

# Hyperparameters
learning_rate = 0.001  # Step size
initial_b = 0.001  # Initial intercept
initial_w = 0.001  # Initial weight
num_iters = 100000  # Number of iterations

# Train the model
train(Y, initial_b, initial_w, X, learning_rate, num_iters)
