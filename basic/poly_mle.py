import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression_mle(X, y, degree):
    # Construct the Vandermonde matrix
    X_poly = np.vander(X.flatten(), degree + 1, increasing=True)

    # Compute MLE estimate of coefficients
    w_mle = np.dot(np.dot(np.linalg.inv(np.dot(X_poly.T, X_poly)), X_poly.T), y)

    return w_mle

# Other functions remain the same
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

# Perform polynomial regression with MLE
w_mle = polynomial_regression_mle(X, y, degree)

# Predict output values
y_pred = predict(X, w_mle)

# Plot the results
plt.scatter(X, y, color='blue', label='Observations')
plt.scatter(X, y_pred, color='red', label='MLE Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with MLE')
plt.legend()
plt.show()


