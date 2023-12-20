import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = 3 * x + 10
y = y_true + np.random.normal(0, 2, x.shape)  # to add some noise


def calculatePredicted_Y(x, alpha, beta):
    return alpha * x + beta



def l0_norm(alpha, beta):
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(errors != 0)



def l1_norm(alpha, beta):
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(np.abs(errors))



def l2_norm(alpha, beta):
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(np.square(errors))



def infinity_norm(alpha, beta):
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.max(np.abs(errors))