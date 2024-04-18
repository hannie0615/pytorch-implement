"""
Assignment 01
Regression 과제 (기계학습 2024-1)

author : 신한이
student ID : 2024021072
goal : To write computer programs of MLE, MAP, and Bayesian methods for predicting
a continuous target variable t for a test sample x

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from generate_data import generate_data

# 1. 데이터 x, t 100개씩 생성
x, t = generate_data(num=100)

# 2. likelihood function 작성
def calcLogLikelihood(guess, true, n):
    error = true-guess
    sigma = np.std(error)
    f = ((1.0/(2.0*math.pi*sigma*sigma))**(n/2))* \
        np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)

def myFunction(weight):
    yGuess = weight[0]
    for i in range(1, 9):
        yGuess += weight[i]*pow(x, i)
    f = calcLogLikelihood(yGuess, t, float(len(yGuess)))

    return (-1*f)



# 3. Regression 수행 (최적화 BFGS 사용)
weight = np.zeros(9)
res = minimize(myFunction, weight, method='BFGS')


print(res)
guess = res['x'][0]
for i in range(1, 9):
    guess += res['x'][i] * pow(x, i)

print(t[0:10])
print(guess[0:10])

plt.plot(x, t, 'ob')
plt.plot(x, guess, 'or')
plt.show()
