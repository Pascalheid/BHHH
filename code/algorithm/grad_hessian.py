# Test field minimization
import numpy as np
import scipy.stats as stats
from scipy.optimize._numdiff import approx_derivative

# Generate data
X = np.random.randn(100, 10) * np.random.uniform(0.5, 4, (1, 10)) + \
    np.random.uniform(-20, 20, (1, 10))

b_true = np.random.randn(10, 1)
b_start = np.zeros((10, 1))
Y = X.dot(b_true) + np.random.randn(100, 1)
data = np.hstack((Y, X))

# Mögliche Beispiele sind Logit, OLS, etc.

# Define quadratic residual function
def res(b, data):
    """
    Calculate the squared residual for an observation and a given feature and 
    parameter vector.
    
    Parameters
    ----------    
    b : numpy (p x 1) array 
        containg the coefficient values
        
    data : ndarray
        Array where the first column contains the dependent variable, while the
        rest of the array contains independent variables.

    Returns
    -------
    The individual loss function value

    """
    
    return ((data[0] - data[1:].dot(b))** 2)


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
    args, kwargs : tuple and dict, optional
        Additional function inputs. args is empty by default while kwargs 
        contains the data array.
    
    Returns
    -------
    grad_array : array
        The partial derivatives of f w.r.t. x0 for every observation.
        

    """
    
    # Calculate function values for each observation
    grad_array = np.empty((nobs, N))
    kwargs_temp = kwargs.copy()
        
    for i in range(nobs):
        
        data_i = {"data" : data[i]}
        kwargs_temp.update(data_i)
        # Get more general type of indexing
        grad_array[i, :] = approx_derivative(fun, 
                                             x0, 
                                             args = args, 
                                             kwargs = kwargs_temp)
                                                        
    return grad_array

# Test
grads = approx_fprime_ind(res, np.empty(10), data = data)

# Hessian matrix approximation
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
    
    outer_pdt = np.empty((nobs, N, N))
    
    for i in range(nobs):
        
        outer_pdt[i, :, :] = np.outer(grad_array[i], grad_array[i])
        
    return outer_pdt.sum(axis = 0) / nobs

app_hess = approx_hess_bhhh(grads)

def aggregate_fun(fun, x0, data, args = (), kwargs = {}):
    """
    Calculate the sum of the function given for every 

    Parameters
    ----------
    fun : callable fun(x, data, *args, **kwargs)
        Objective function to aggregate over.
    x0 : ndarray    
        Parameters for which to calculate the function values.
    data : ndarray
        Array containing the data over which to aggregate.
    args : tuple, optional
        Extra arguments passed to fun. The default is ().
    kwargs : dict, optional
        Extra arguments passed to fun. The default is {}.

    Returns
    -------
    fun_agg : callable fun(x, *args)
        Sum of the objective function evaluations

    """
    
    fun_agg = np.empty(nobs)
          
    for i in range(nobs):
        
        fun_agg[i] = fun(x0, data = data[i], *args, **kwargs)
        
    return fun_agg.sum()
            
# Define normal density for regression
x = np.random.normal(5, 2, 100)
def neg_log_dnorm(theta, data):
    
    return - np.log(stats.norm.pdf(data, theta[0], theta[1]))

def neg_log_dnorm_ols(theta, data):
    
    return - np.log(stats.norm.pdf((data[:, 0] - theta[1:].dot(data[:, 1:])) ** 2, 
                    [0], theta[0]))