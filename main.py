import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = 3 * x + 10
y = y_true + np.random.normal(0, 2, x.shape)  # to add some noise