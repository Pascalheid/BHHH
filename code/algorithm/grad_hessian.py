# Test field minimization
import numpy as np
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
    fun : callable
        loss function.
    
    x0 : ndarray
        Parameters for which we want to calculate the derivative.
    
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
        
    for i in range(nobs):
        
        # Get more general type of indexing
        grad_array[i, :] = approx_derivative(fun, 
                                             x0, 
                                             args = args, 
                                             kwargs = kwargs.update(
                                                 {"data" : data[i, :]}))
                                                
    return grad_array

# Test
grad(res, np.empty(10), data = data)

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
        
        outer_pdt[i, :, :] = np.outer(grad_array[i, :], grad_array[i, :])
        
    return outer_pdt.sum(axis = 0) / nobs

# function to calculate sum
def aggregate_fun(fun, data, args = (), kwargs = {}):
    """
    Calculate the sum of the function given for every 

    Parameters
    ----------
    fun : callable fun(x, *args)
        Objective function to aggregate over.
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
        
        fun_agg[i] = fun(data = data[i, :], *args, **kwargs)
        
    return fun_agg.sum()

def aggregate_fprime(grad_array):
    """
    Sum over the individual gradients of the function given to 
    approx_fprime_ind.

    Parameters
    ----------
    grad_array : ndarray
        Output of the function approx_fprime_ind.

    Returns
    -------
    grad_aggregate : ndarray
        Sum of the individual gradient evaluations

    """
    
    return grad_array.sum(axis = 0)
    

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
        
        