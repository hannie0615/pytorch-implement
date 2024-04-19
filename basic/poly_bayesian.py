import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression_bayesian(X, y, degree, n_samples=2000, burn_in=1000, proposal_sd=1):
    """
    Perform polynomial regression with a fully Bayesian approach using Metropolis-Hastings MCMC.

    Parameters:
    X : array-like, shape (n_samples, 1)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    degree : int
        Degree of the polynomial.
    n_samples : int, optional
        Number of MCMC samples to draw (default is 2000).
    burn_in : int, optional
        Number of burn-in samples (default is 1000).
    proposal_sd : float, optional
        Standard deviation of the proposal distribution

    Returns:
    samples : array-like, shape (n_samples - burn_in, degree+1)
        Samples from the posterior distribution of polynomial coefficients.
    """
    # Initialize coefficients randomly
    coeffs_current = np.random.normal(0, 1, degree + 1)

    # Acceptance counter
    accept_count = 0

    # Store samples
    samples = []

    for _ in range(n_samples + burn_in):
        # Propose new coefficients
        coeffs_proposed = coeffs_current + np.random.normal(0, proposal_sd, degree + 1)

        # Compute log-likelihood for proposed coefficients
        y_pred_proposed = np.dot(np.vander(X.flatten(), degree + 1, increasing=True), coeffs_proposed)
        log_likelihood_proposed = -0.5 * np.sum((y - y_pred_proposed) ** 2)

        # Compute log-likelihood for current coefficients
        y_pred_current = np.dot(np.vander(X.flatten(), degree + 1, increasing=True), coeffs_current)
        log_likelihood_current = -0.5 * np.sum((y - y_pred_current) ** 2)

        # Compute log-prior (assuming normal prior with mean 0 and variance 1)
        log_prior_proposed = -0.5 * np.sum(coeffs_proposed ** 2)
        log_prior_current = -0.5 * np.sum(coeffs_current ** 2)

        # Compute acceptance probability
        log_acceptance_prob = (log_likelihood_proposed + log_prior_proposed) - (log_likelihood_current + log_prior_current)

        # Accept or reject the proposal
        if np.log(np.random.rand()) < log_acceptance_prob:
            coeffs_current = coeffs_proposed
            accept_count += 1

        # Store sample if burn-in period is over
        if _ >= burn_in:
            samples.append(coeffs_current)

    print("Acceptance rate:", accept_count / (n_samples + burn_in))

    return np.array(samples)


# Generate some sample data
from generate_data import generate_data
X, y = generate_data(num=100)

# Define the degree of the polynomial
degree = 9

# Perform polynomial regression with Bayesian approach
samples = polynomial_regression_bayesian(X, y, degree)

# Plot posterior predictive samples
for sample in samples:
    y_pred = np.dot(np.vander(X.flatten(), degree + 1, increasing=True), sample)
    plt.scatter(X, y_pred, color='red', alpha=0.1)

plt.scatter(X, y, color='blue', label='Observations')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Bayesian Approach')
plt.show()
