# Test field minimization
import numpy as np
import scipy.stats as stats
from numba import jit
from scipy.optimize._numdiff import approx_derivative

# Mögliche Beispiele sind Logit, OLS, etc.
# Calculate the gradient of the loss
def approx_fprime_ind(fun, x0, data, args = (), kwargs = {}):
    """
    Calculate the gradient with respect to the coefficient vector b for every
    row in the data array.

    Parameters
    ----------
    fun : callable fun(x, data, *args, **kwargs)
        loss function.
    x0 : ndarray
        Parameters for which we want to calculate the derivative.
    epsilon : int or ndarray, optional
        If fprime is approximated, use this value for the step size.
    data : ndarray
        Data on which to calculate the likelihood function.
    nobs : int, optional
        Argument that passes the number of rows in a dataset.
    args, kwargs : tuple and dict, optional
        Additional function inputs. args is empty by default while kwargs 
        contains the data array.
    
    Returns
    -------
    grad_array : array
        The partial derivatives of f w.r.t. x0 for every observation.
        

    """
    kwargs_temp = kwargs.copy()
    
    try:
        kwargs_temp.update({"data" : data})
        grad_array = approx_derivative(fun, x0, kwargs = kwargs_temp)
        return grad_array
    except IndexError:
        print("Function {} couldn't be vectorized.".format(fun))        
                                                        
    return grad_array


# Hessian matrix approximation
@jit(nopython = True)
def approx_hess_bhhh(grad_array):
    """
    Approximating the Hessian matrix by calculating the sum of the outer 
    products of the gradients evaluated at each individual observation.

    Parameters
    ----------
    grad_array : ndarray
        An array containing the gradient of the objective function for every 
        individual.

    Returns
    -------
    Hessian : ndarray
        Approximated Hessian matrix resulting from the outer product of the
        gradients.

    """        
   
    nobs = data.shape[0]
    N = grad_array.shape[1]
    
    outer_pdt = np.empty((nobs, N, N))
    
    for i in range(nobs):
        
        outer_pdt[i, :, :] = np.outer(grad_array[i], grad_array[i])
        
    return outer_pdt.sum(axis = 0) / nobs
            
# Define normal density for regression
x = np.random.normal(5, 2, 10000)
def neg_log_dnorm(theta, data):
    
    return - np.log(stats.norm.pdf(data, theta[0], theta[1]))

def neg_log_dnorm_ols(theta, data):
    
    nobs = data.shape[0]
    
    return (nobs / 2 * np.log(2 * np.pi * theta[0]) +
        0.5 * (data[:, 0] - data[:, 1:].dot(theta[1:]) ** 2 / theta[0]))

# Example
# Generate data
X = np.random.randn(10000, 10) * np.random.uniform(0.5, 4, (1, 10)) + \
    np.random.uniform(-20, 20, (1, 10))

b_true = np.random.randn(10, 1)
Y = X.dot(b_true) + np.random.randn(10000, 1)
data = np.hstack((Y, X))

theta_true = np.array([1] + b_true.flatten().tolist())
theta_zero = np.ones(11)
neg_log_dnorm_ols(theta, data)

def neg_log_binary_logistic(theta, data):
    
    nobs = data.shape[0]
    
    
# Simulated dataset
Y_cat = Y >= Y.mean()

