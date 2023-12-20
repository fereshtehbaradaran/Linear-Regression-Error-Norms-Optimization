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

# Modify the functions to accept a single parameter
def l0_norm(params):
    alpha, beta = params
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(errors != 0)

def l1_norm(params):
    alpha, beta = params
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(np.abs(errors))

def l2_norm(params):
    alpha, beta = params
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.sum(np.square(errors))

def infinity_norm(params):
    alpha, beta = params
    errors = calculatePredicted_Y(x, alpha, beta) - y
    return np.max(np.abs(errors))

# Optimization
initial_params = [0, 0]  # Initial guess for alpha and beta

result_l1 = minimize(l1_norm, initial_params, method='SLSQP')
result_l2 = minimize(l2_norm, initial_params, method='SLSQP')
result_inf = minimize(infinity_norm, initial_params, method='SLSQP')

print(result_l1, result_l2, result_inf)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data points', marker='x', color='red')
plt.plot(x, y_true, label='True line', color='black', linestyle='--')

plt.plot(x, calculatePredicted_Y(x, *result_l1.x), label='L1 Norm Regression')
plt.plot(x, calculatePredicted_Y(x, *result_l2.x), label='L2 Norm Regression')
plt.plot(x, calculatePredicted_Y(x, *result_inf.x), label='Infinity Norm Regression')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Regression Lines for Different Norms')
plt.legend()
plt.show()