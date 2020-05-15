import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import statsmodels.api as sm

from src.algorithms.fmin_bbhhh import fmin_bhhh
from src.algorithms.fmin_l_bfgs_b import fmin_l_bfgs_b

# Define normal density for regression
# Set seed
np.random.seed(100)

x = np.random.normal(5, 2, 10000)  # Mean = 5, SD = 2


def neg_log_dnorm(theta, data):

    return -np.log(stats.norm.pdf(data, theta[0], theta[1]))


def neg_log_lk_ols(theta, data):

    first_term = 0.5 * np.log(2 * np.pi * theta[0])
    second_term = 0.5 * (data[:, 0] - data[:, 1:].dot(theta[1:])) ** 2 / theta[0]
    return first_term + second_term


# Example
# Generate data
np.random.seed(100)
X = np.random.randn(10000, 2) * np.random.uniform(0.5, 4, (1, 2)) + np.random.uniform(
    -20, 20, (1, 2)
)
b_true = np.random.randn(2, 1)
Z = X.dot(b_true)
Pr = 1 / (1 + np.exp(-Z))
y = np.random.binomial(1, Pr)
data = np.hstack((y, X))

# OLS example
Y = X.dot(b_true) + np.random.randn(10000, 1)
data = np.hstack((Y, X))

theta_true = np.array([1] + b_true.flatten().tolist())
theta_zero = np.ones(3)


def neg_log_binary_logistic(theta, data):

    return -(
        data[:, 0] * data[:, 1:].dot(theta) - np.log(1 + np.exp(data[:, 1:].dot(theta)))
    )


# Simulated dataset


# L_bfgs_b function takes the sum of the individual log likelihoods as input
def sum_neg_log_bin_logistic(x0):
    return neg_log_binary_logistic(x0, data).sum()


# Starting point (1, 1)
x0 = np.ones(2)
res = fmin_bhhh(fun=neg_log_binary_logistic, x0=x0, args=data)
fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0)
sm.Logit(data[:, 0], data[:, 1:]).fit().summary()
opt.fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, approx_grad=1)

# Constrained optimization
# Non binding
bounds = np.array([[-5, -5], [5, 5]])  # Lower Bound  # Upper Bound

# Report number of iterations
fmin_bhhh(neg_log_binary_logistic, x0, bounds, args=data)
fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, bounds=bounds)
opt.fmin_l_bfgs_b(
    sum_neg_log_bin_logistic, x0, approx_grad=1, bounds=[(-5, 5), (-5, 5)]
)

# Binding
bounds = np.array([[1, 0], [5, 5]])  # Lower Bound  # Upper Bound

# Report number of iterations
fmin_bhhh(neg_log_binary_logistic, x0, bounds, args=data)
fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, bounds=bounds)
opt.fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, approx_grad=1, bounds=[(1, 5), (0, 5)])
