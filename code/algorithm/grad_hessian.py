# Test field minimization
import numpy as np
import scipy as sp
from scipy.optimize._numdiff import approx_derivative

# Generate data
X = np.random.randn(100, 10) * np.random.uniform(0.5, 4, (1, 10)) + \
    np.random.uniform(-20, 20, (1, 10))

b_true = np.random.randn(10, 1)
b_start = np.zeros((10, 1))
Y = X.dot(b_true) + np.random.randn(100, 1)

# Mögliche Beispiele sind Logit, OLS, etc.

# Define quadratic residual function
def res(b, y, x):
    """
    Calculate the squared residual for an observation and a given feature and 
    parameter vector.
    
    Parameters
    ----------
    y : scalar containing the dependent variable
    
    x : numpy (1 x p) array containing the independent variables
    
    b : numpy (p x 1) array containg the coefficient values

    Returns
    -------
    The individual loss function

    """
    
    return ((y - x.dot(b))** 2)[0]

def sum_res(b, Y, X):
    """
    Return the sum of squared residuals

    Parameters
    ----------
    b : numpy array (p x 1) containing the coefficient values.
    
    Y : numpy array (n x 1) containing the dependent variable's values.
    
    X : numpy array (n x p) containing the independent variable's values.

    Returns
    -------
    nfloat: Sum of squared residuals.

    """
    loss = np.empty(len(Y))   # n dimensional vector
    
    for i in range(len(Y)):
        
        loss[i] = res(b, Y[i, :], X[i, :])
        
    return loss.sum()
    
# Calculate the gradient of the loss
def grad(f, b, Y, X):
    """
    Calculate the gradient with respect to the coefficient vector b.

    Parameters
    ----------
    f : callable
        loss function.
    
    b : array
        Parameters for which we want to calculate the derivative.
    
    Y : array
        The dependent variable
    
    X : array
        The indipendent variables.

    Returns
    -------
    grad: array
        The partial derivatives of f w.r.t. b for every observation in X and Y.

    """
    
    # Calculate function values for each observation
    grad_array = np.empty(X.shape)
        
    for i in range(len(Y)):
        
        grad_array[i, :] = approx_derivative(f, b, args = (Y[i, :], X[i, :]))
        
    return grad_array, grad_array.sum(axis = 0)

# Test
grad(res, b_start.reshape(10), Y, X)

# Hessian matrix approximation
def hess_approx(gradient):
    """
    

    Parameters
    ----------
    gradient : ndarray
        An array containing the gradient of the objective function for every 
        individual.

    Returns
    -------
    Hessian : ndarray
        Approximated Hessian matrix resulting from the outer product of the
        gradients.

    """
    
    outer_pdt = np.empty((100, 10, 10))
    
    for i in range(gradient.shape[0]):
        
        outer_pdt[i, :, :] = np.outer(gradient[i, :], gradient[i, :])
        
    return outer_pdt.sum(axis = 0) / gradient.shape[0]

# Implement algorithm
# End the algorithm if the norm of the gradient is smaller than a certain 
# threshold i.e. 1e-05.
# Order of the norm
# Maximum iterations
# Still need line search algorithm
# Option to add fprime as a functional if known
def fmin_bhhh(f, x0, args = (), gtol = 1e-05, norm = 2, maxiter = 10000):
    """
    

    Parameters
    ----------
    f : callable f(x, *args)
        Objective function to be minimized.
    
    x0 : ndarray
        Initial guess.
    
    fprime : callable f'(x, *args), optional
        Gradient of f.
    
    args : tuple, optional
        Etra arguments passed to f and fprime. Default is ().
    
    gtol : float, optional
        Gradient norm must be less than gtol before successful termination. 
        Default value is 1e-05.
    
    norm : float, optional
        Order of norm (Inf is max, - Inf is min).
    
    maxiter: float, optional
        Maximum number of iterations to perform.

    Returns
    -------
    xopt: ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
        
    fopt: float
        Minimum value.

    """
    
    # Setup empty numpy array
    x_update = x0
    
    # Calculate 
    for i in range(maxiter):
        
        # Calculate gradient and Hessian at previous guess
        delta_g = grad(f, x_update[i].reshape, *args)
        
        # Check that gradient is below threshold
        if np.linalg.norm(delta_g, norm) < gtol:
            
            break
        
        gradobjfct = grad_obj_fct(delta_g)
        Hessian = hess_approx(delta_g)
        
        # Update step
        x_update.concatenate(x_update[i] - 
                             np.linalg.inv(Hessian).dot(gradobjfct), 
                             axis = 0)
        
    # Return fopt
    return x_update[x_update.shape[0]]
        
        