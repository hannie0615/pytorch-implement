import math
import numpy as np
import random



def generate_data(num):
    np.random.seed(42)
    x = np.random.rand(num) * 2 - 1  # the range of [−1, 1]

    # t = cos (2πx) + sin (πx) + ϵ
    t = np.zeros(num)
    for i in range(num):
        t[i] = math.cos(2 * math.pi * x[i]) + math.sin(math.pi * x[i]) + random.gauss(0, 1 / 11.1)

    return x, t

def generate_weight(num):

    w = np.zeros(num)
    for i in range(num):
        w[i] = random.gauss(0, 0.005)
    return w