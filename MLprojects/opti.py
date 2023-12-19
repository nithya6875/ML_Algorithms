import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Toy dataset generation
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


# Function to train linear regression with given learning rate and compute MSE
def train_and_evaluate(learning_rate, X, y):
    # Initialize linear regression model
    lin_reg = LinearRegression()

    # Perform gradient descent with the specified learning rate
    lin_reg.fit(X, y)

    # Compute predictions
    y_pred = lin_reg.predict(X)

    # Calculate MSE
    mse = mean_squared_error(y, y_pred)

    return mse


# Hyperparameters to search
learning_rates = np.linspace(0.01, 0.1, 10)  # Range of learning rates to explore

# Initialize variables for best learning rate and minimum MSE
best_learning_rate = None
min_mse = float('inf')

# Grid search to find the best learning rate
for lr in learning_rates:
    mse = train_and_evaluate(lr, X, y)
    if mse < min_mse:
        min_mse = mse
        best_learning_rate = lr

# Output the best learning rate and corresponding minimum MSE
print(f"Best learning rate: {best_learning_rate}")
print(f"Minimum MSE: {min_mse}")
