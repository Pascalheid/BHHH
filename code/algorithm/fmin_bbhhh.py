# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:03:05 2020

@author: Wilms
"""
import numpy as np
from grad_hessian_vectorized import approx_fprime_ind, approx_hess_bhhh  
from scipy.optimize.optimize import _check_unknown_options, _epsilon, \
    _line_search_wolfe12, _status_message, OptimizeResult, vecnorm, \
    _LineSearchError 


# Minimization function.
def fmin_bhhh(fun, x0, data, bounds = None, fprime = None, args = (), 
              kwargs = {} , tol = {"abs" : 1e-05, "rel" : 1e-08}, 
              norm = np.Inf, epsilon = _epsilon, maxiter = None, 
              full_output = 0, disp = 1, retall = True, callback = None):
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
    epsilon : int or ndarray, optional
        If fprime is approximated, use this value for the step size.
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

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'BFGS' `method` in particular.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Berndt, Hall, Hall,
    and Hubert (BHHH)

    References
    ----------
    REFERENCE NEEDED.

    """
    opts = {'tol': tol,
            'norm': norm,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_bhhh(fun, x0, data, bounds, args, kwargs, fprime, 
                         callback = callback, 
                         **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_bhhh(fun, x0, data, bounds = None, args = (), kwargs = {}, 
                   jac = None, callback = None, 
                   tol = {"abs" : 1e-05, "rel" : 1e-08}, norm = np.Inf, 
                   maxiter = None, disp = False, return_all = False, 
                   **unknown_options):
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
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)
    
    f = fun
    fprime = jac
    # epsilon = eps Add functionality
    retall = return_all
    k = 0
    N = len(x0)
    nobs = data.shape[0]
    
    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    
    if bounds is None:
        bounds = np.array([np.inf] * N * 2).reshape((2, N))
        bounds[0, :] = - bounds[0, :]
    if bounds.shape[1] != N:
        raise ValueError('length of x0 != length of bounds')
    
    low = bounds[0, :]
    up = bounds[1, :]
    x0 = np.clip(x0, low, up)
    
    if maxiter is None:
        maxiter = len(x0) * 200
    
    # Need the aggregate functions to take only x0 as an argument  
    agg_fun = lambda x0 : f(x0, data, *args, **kwargs).sum()

    if not callable(fprime):
        myfprime = approx_fprime_ind
    else:
        myfprime = fprime

    agg_fprime = lambda x0 : myfprime(f, x0, data, args, kwargs).sum(axis = 0)

    # Setup for iteration
    old_fval = agg_fun(x0)
    
    gf0 = agg_fprime(x0)
    norm_pg0 = vecnorm(x0 - np.clip(x0 - gf0, low, up), ord = norm)
    
    xk = x0
    norm_pgk = norm_pg0
    
    if retall:
        allvecs = [x0]
    warnflag = 0

    for i in range(maxiter): # for loop instead.
        
        # Calculate indices ofactive and inative set using projected gradient
        epsilon = min(np.min(up - low) / 2, norm_pgk)
        activeset = np.logical_or(x0 - low <= epsilon, up - x0 <= epsilon)
        inactiveset = np.logical_not(activeset)
        
        # Individual
        gfk_obs = myfprime(f, xk, data, args, kwargs)
        
        # Aggregate fprime. Might replace by simply summing up gfk_obs
        gfk = gfk_obs.sum(axis = 0)
        norm_pgk = vecnorm(xk - np.clip(xk - gfk, low, up), ord = norm)
        
        # Check tolerance of gradient norm
        if(norm_pgk <= tol["abs"] + tol["rel"] * norm_pg0):
            break

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2
        
        # Calculate BHHH hessian and step
        Hk = approx_hess_bhhh(gfk_obs[:, inactiveset])  # Yes
        Bk = np.linalg.inv(Hk)
        pk = np.empty(N)
        pk[inactiveset] = - np.dot(Bk, gfk[inactiveset])
        pk[activeset] = - gfk[activeset]
       
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                      _line_search_wolfe12(agg_fun, 
                                          agg_fprime, 
                                          xk, pk, gfk,
                                          old_fval, old_old_fval, 
                                          amin = 1e-100, amax = 1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = np.clip(xk + alpha_k * pk, low, up)
        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if callback is not None:
            callback(xk)
        k += 1

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

    fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        
    result = OptimizeResult(fun = fval, jac = gfk, hess_inv = Hk, 
                            status = warnflag,
                            success = (warnflag == 0), 
                            message = msg, x = xk, nit = k)
    if retall:
        result['allvecs'] = allvecs
    return result
