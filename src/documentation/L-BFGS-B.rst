.. _L-BFGS-B:

**********
L-BFGS-B
**********

""""""""""""""
Wrap Function
""""""""""""""

This is the wrapping function.

.. automodule:: src.algorithms.fmin_l_bfgs_b
    :members: fmin_l_bfgs_b

""""""""""""""
Algorithm
""""""""""""""

This function performs the Quasi Newton-Rhapson algorithm with line search using the Hessian calculated
via the L-BFGS algorithm.

.. automodule:: src.algorithms.fmin_l_bfgs_b
    :members: _minimize_lbfgsb

""""""""""""""
Hessian Calculation
""""""""""""""

In this function Hessian is calculated throught the L-BFGS algorithm.

.. automodule:: src.algorithms.grad_hessian_vectorized
    :members: bfgsrecb


""""""""""""""
Gradient Approximation
""""""""""""""

This function approximates the gradient of the objective funtion if it is not given as an input to *fmin_l_bfgs_b*.

.. automodule:: src.algorithms.grad_hessian_vectorized
    :members: approx_fprime_ind
