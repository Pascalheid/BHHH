# flake8: noqa
import numpy as np
from numba import jit
from scipy.optimize._numdiff import approx_derivative

# Mögliche Beispiele sind Logit, OLS, etc.
# Calculate the gradient of the loss


def approx_fprime_ind(fun, x0, data, args=(), kwargs={}):

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
        kwargs_temp.update({"data": data})
        grad_array = approx_derivative(fun, x0, kwargs=kwargs_temp)
        return grad_array
    except IndexError:
        print(f"Function {fun} couldn't be vectorized.")

    return grad_array


# Hessian matrix approximation
@jit(nopython=True)
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

    nobs = grad_array.shape[0]
    N = grad_array.shape[1]

    outer_pdt = np.empty((nobs, N, N))

    for i in range(nobs):

        outer_pdt[i, :, :] = np.outer(grad_array[i], grad_array[i])

    return outer_pdt.sum(axis=0)


def bfgsrecb(nt, sstore, ystore, pk, activeset):
    """
    Recursively compute action of an approximated Hessian on a vector using
    stored information of the history of the iteration

    Parameters
    ----------
    nt : int
        The number of saved iterations.
    sstore, ystore : ndarray
        Vector used to calculate the approximation to the Hessian.
    pk : ndarray
        Current descent direction.
    activeset : logical ndarray
        Array indicating for which variables the constraints are binding.

    Returns
    -------
    pk : ndarray
        Returns the step size.

    """

    pk[activeset] = 0
    if nt == 0:
        return pk

    sstore[nt - 1, :][activeset] = 0
    ystore[nt - 1, :][activeset] = 0

    Alpha = sstore[nt - 1, :].dot(pk)
    pk = pk - Alpha * ystore[nt - 1, :]

    newset = np.int_([])
    pk = bfgsrecb(nt - 1, sstore, ystore, pk, newset)

    pk = pk + (Alpha - ystore[nt - 1, :].dot(pk)) * sstore[nt - 1, :]
    pk[activeset] = 0

    return pk
