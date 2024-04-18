import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from generate_data import *

# 1. 데이터 x, t 100개씩 생성
x, t = generate_data(num=100)
log_prior = generate_weight(num=100)
print(log_prior)
def calcLogLikelihood(guess, true, n):

    error = true-guess
    sigma = np.std(error)
    f = ((1.0/(2.0*math.pi*sigma*sigma))**(n/2))*np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)

def myFunction(weight):

    yGuess = weight[0]
    for i in range(1, 9):
        yGuess += weight[i]*pow(x, i)
    f = calcLogLikelihood(yGuess, t, float(len(yGuess)))
    f += np.log(log_prior)

    return (-1*f)



# 3. Regression 수행 (최적화 BFGS 사용)
weight = np.zeros(9)
res = minimize(myFunction, weight, method='BFGS')
guess = res['x'][0]
for i in range(1, 9):
    guess += res['x'][i] * pow(x, i)

print(t[0:10])
print(guess[0:10])

plt.plot(x, t, 'ob')
plt.plot(x, guess, 'or')
plt.show()


# data = x
# # Define the prior distribution (another normal distribution) as prior knowledge
# prior_mean = 0
# prior_std = 0.005
#
#
# def posterior(x):
#     # Likelihood function (normal distribution)
#     likelihood = norm.pdf(data, x, 1)
#
#     # Prior probability density function (normal distribution)
#     prior = norm.pdf(x, prior_mean, prior_std)
#
#     # Posterior probability proportional to likelihood * prior
#     posterior_prop = likelihood * prior
#
#     # posterior = posterior_prop / np.sum(posterior_prop)
#
#     return posterior_prop
#
#
# # Use optimization to find the value of x that maximizes the posterior
# map_estimate = np.argmax(posterior(t))  # Adjust range for better fit
#
# print("MAP estimate:", map_estimate)