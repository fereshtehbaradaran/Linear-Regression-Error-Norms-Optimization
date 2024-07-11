# Linear Regression Error Norms Optimization

## Introduction

This repository contains the implementation of a Python program designed to minimize various error norms in the context of linear regression. Linear regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables. This project focuses on finding the best fit line through a set of data points and evaluates the fit based on different error norms to provide a comprehensive view of model accuracy.

## Features

The program performs the following key tasks:

1. **Data Generation**
2. **Linear Model and Error Calculation**
3. **Error Norm Minimization**
    - L0 Norm: Counts the number of non-zero errors.
    - L1 Norm: Sum of absolute values of errors.
    - L2 Norm: Sum of squares of errors.
    - Infinity Norm: Maximum absolute error.
4. **Visualization of Results**

## Implementation Details

### Data Generation

The dataset is generated using the numpy library. A linear relationship is simulated with added Gaussian noise to mimic real-world data variability.

### Linear Model and Error Calculation

The linear model is defined as a function, `calculatePredicted_Y`, which calculates \( y \) based on input \( x \), and model parameters \( \alpha \) and \( \beta \). The signed error is then calculated for each data point.

### Error Norm Minimization

Each error norm (L0, L1, L2, and Infinity) is minimized using a specific methodology:

#### L0 Norm Minimization

- **Methodology**: Counts the number of non-zero errors in the prediction. Minimization is achieved through an iterative approach.
- **Code Explanation**: The program defines a function to calculate the L0 error for given \( \alpha \) and \( \beta \), then iteratively searches over a range of values to minimize this count.

#### L1 Norm Minimization

- **Methodology**: Sum of absolute values of errors. Minimized using an iterative approach.
- **Code Explanation**: The function for L1 error calculates the sum of absolute differences between actual and predicted values, then iteratively explores different values to minimize the total absolute error.

#### L2 Norm Minimization

- **Methodology**: Least squares error, minimized using gradient descent.
- **Code Explanation**: The function for L2 error calculates the sum of squared differences, and the gradient descent algorithm adjusts parameters proportionally to the negative gradient to reduce the L2 error.

#### Infinity Norm Minimization

- **Methodology**: Focuses on the maximum absolute error. Minimization is done iteratively.
- **Code Explanation**: The function for the Infinity norm error finds the maximum absolute error given \( \alpha \) and \( \beta \), then iteratively tests different values to minimize this maximum error.

### Visualization

The dataset and the resulting regression lines for each error norm are visualized using `matplotlib.pyplot`.


## Getting Started

### Prerequisites

- Python 3.x
- numpy
- matplotlib

### Installation

Clone the repository:
```sh
git clone https://github.com/fereshtehbaradaran/Linear-Regression-Error-Norms-Optimization.git
```

### Usage
Run the main script to generate data, minimize error norms, and visualize results:
```sh
python main.py
```
