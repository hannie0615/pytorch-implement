import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def polynomial_regression_map(X, y, degree, alpha):
    # Construct the Vandermonde matrix
    X_poly = np.vander(X.flatten(), degree + 1, increasing=True)

    # Compute MAP estimate of coefficients
    I = np.eye(degree + 1)
    w_map = np.dot(np.dot(inv(alpha * I + np.dot(X_poly.T, X_poly)), X_poly.T), y)

    return w_map

def predict(X, w):
    degree = len(w) - 1
    X_poly = np.vander(X.flatten(), degree + 1, increasing=True)
    y_pred = np.dot(X_poly, w)
    return y_pred

# Generate some sample data
from generate_data import generate_data
X, y = generate_data(num=100)

# Define the degree of the polynomial
degree = 9

# Define the regularization parameter
alpha = 0.005

# Perform polynomial regression with MAP estimation
w_map = polynomial_regression_map(X, y, degree, alpha)

# Predict output values
y_pred = predict(X, w_map)

# Plot the results
plt.scatter(X, y, color='blue', label='Observations')
plt.scatter(X, y_pred, color='red', label='MAP Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with MAP Estimation')
plt.legend()
plt.show()