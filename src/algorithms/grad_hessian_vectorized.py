# flake8: noqa
import numpy as np
from scipy.optimize._numdiff import approx_derivative


def wrap_function_agg(function, args):
    """
    Wrap function for the objective function in the BHHH.

    """
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args)).sum()

    return ncalls, function_wrapper


def wrap_function_num_dev(objective_fun, args):
    """
    Wrap function for the numerical Jacobian in the BHHH.

    """
    ncalls = [0]

    def function_wrapper(x0):
        ncalls[0] += 1
        return approx_derivative(objective_fun, x0, args=args)

    return ncalls, function_wrapper


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
