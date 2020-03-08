# flake8: noqa
import numpy as np
from grad_hessian_vectorized import bfgsrecb
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize.optimize import _check_unknown_options
from scipy.optimize.optimize import _line_search_wolfe12
from scipy.optimize.optimize import _LineSearchError
from scipy.optimize.optimize import _status_message
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize.optimize import vecnorm


# Minimization function.
def fmin_l_bfgs_b(
    fun,
    x0,
    bounds=None,
    fprime=None,
    args=(),
    kwargs={},
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    maxiter=None,
    full_output=0,
    disp=1,
    retall=True,
    callback=None,
):
    """
    Minimize a function using the BHHH algorithm.

    Parameters
    ----------
    fun : callable fun(x, data, *args, **kwargs)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args), optional
        Gradient of f.
    data : ndarray
        Data points on which to fit the likelihood function to.
    bounds : ndarray, optional
        ``(min, max)`` pairs for each element along the rows``x``, defining
        the bounds on that parameter. Use +-inf for one of ``min`` or
        ``max`` when there is no bound in that direction.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    tol : dict, optional
        Dict that contains the absolute and relative tolerance parameters.
        Form should be tol = {"abs" : x, "rel" : y}. Both parameters must be
        strictly positive and relative tolerance must be bigger equal to the
        absolute tolerance.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True,return fopt, func_calls, grad_calls, and warnflag
        in addition to xopt.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs  :  list
        The value of xopt at each iteration.  Only returned if retall is True.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Berndt, Hall, Hall,
    and Hubert (BHHH). Box constraints are implemented by using a simple
    gradient approach at each step to identify active and inactive variables.
    The standard BHHH approach is then used on the inactive subset.

    References
    ----------
    Berndt, E.; Hall, B.; Hall, R.; Hausman, J. (1974). "Estimation and
    Inference in Nonlinear Structural Models". Annals of Economic and Social
    Measurement. 3 (4): 653–665.
    Buchwald, S. "Implementierung des L-BFGS-B-Verfahrens in Python".
    Bachelor-Thesis University of Konstanz.

    """
    opts = {
        "tol": tol,
        "norm": norm,
        "disp": disp,
        "maxiter": maxiter,
        "return_all": retall,
    }

    res = _minimize_lbfgsb(
        fun, x0, bounds, args, kwargs, fprime, callback=callback, **opts
    )

    return res


def _minimize_lbfgsb(
    fun,
    x0,
    bounds=None,
    args=(),
    kwargs={},
    jac=None,
    callback=None,
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    maxiter=None,
    disp=False,
    return_all=False,
    **unknown_options
):
    """
    Minimization of scalar function of one or more variables using the
    BHHH algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    tol : dict
        Absolute and relative tolerance values.
    norm : float
        Order of norm (Inf is max, -Inf is min).

    """
    _check_unknown_options(unknown_options)

    def f(x0):
        return fun(x0, *args, **kwargs)

    fprime = jac
    # epsilon = eps Add functionality
    retall = return_all
    k = 0
    ns = 0
    nsmax = 5
    N = len(x0)

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)

    if bounds is None:
        bounds = np.array([np.inf] * N * 2).reshape((2, N))
        bounds[0, :] = -bounds[0, :]
    if bounds.shape[1] != N:
        raise ValueError("length of x0 != length of bounds")

    low = bounds[0, :]
    up = bounds[1, :]
    x0 = np.clip(x0, low, up)

    if maxiter is None:
        maxiter = len(x0) * 200

    if not callable(fprime):

        def myfprime(x0):
            return approx_derivative(f, x0, args=args, kwargs=kwargs)

    else:
        myfprime = fprime

    # Setup for iteration
    old_fval = f(x0)

    gf0 = myfprime(x0)
    gfk = gf0
    norm_pg0 = vecnorm(x0 - np.clip(x0 - gf0, low, up), ord=norm)

    xk = x0
    norm_pgk = norm_pg0

    sstore = np.zeros((maxiter, N))
    ystore = sstore.copy()

    if retall:
        allvecs = [x0]
    warnflag = 0

    # Calculate indices ofactive and inative set using projected gradient
    epsilon = min(np.min(up - low) / 2, norm_pgk)
    activeset = np.logical_or(xk - low <= epsilon, up - xk <= epsilon)
    inactiveset = np.logical_not(activeset)

    for _ in range(maxiter):  # for loop instead.

        # Check tolerance of gradient norm
        if norm_pgk <= tol["abs"] + tol["rel"] * norm_pg0:
            break

        pk = -gfk
        pk = bfgsrecb(ns, sstore, ystore, pk, activeset)
        gfk_active = gfk.copy()
        gfk_active[inactiveset] = 0
        pk = -gfk_active + pk

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                f,
                myfprime,
                xk,
                pk,
                gfk,
                old_fval,
                old_old_fval,
                amin=1e-100,
                amax=1e100,
            )
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = np.clip(xk + alpha_k * pk, low, up)

        if retall:
            allvecs.append(xkp1)

        yk = myfprime(xkp1) - gfk
        sk = xkp1 - xk
        xk = xkp1
        gfk = myfprime(xkp1)

        norm_pgk = vecnorm(xk - np.clip(xk - gfk, low, up), ord=norm)

        # Calculate indices ofactive and inative set using projected gradient
        epsilon = min(np.min(up - low) / 2, norm_pgk)
        activeset = np.logical_or(xk - low <= epsilon, up - xk <= epsilon)
        inactiveset = np.logical_not(activeset)

        yk[activeset] = 0
        sk[activeset] = 0

        # reset storage
        ytsk = yk.dot(sk)
        if ytsk <= 0:
            ns = 0
        if ns == nsmax:
            print("ns reached maximum size")
            ns = 0
        elif ytsk > 0:
            ns += 1
            alpha0 = ytsk ** 0.5
            sstore[ns - 1, :] = sk / alpha0
            ystore[ns - 1, :] = yk / alpha0

        k += 1

        if callback is not None:
            callback(xk)

        if np.isinf(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break
        if np.isnan(xk).any():
            warnflag = 3
            break

    fval = old_fval

    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        print("{}{}".format("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)

    result = OptimizeResult(
        fun=fval,
        jac=gfk,
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xk,
        nit=k,
    )
    if retall:
        result["allvecs"] = allvecs
    return result
