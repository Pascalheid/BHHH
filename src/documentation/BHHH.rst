.. _BHHH:

**********
BHHH
**********

""""""""""""""
Wrap Function
""""""""""""""

This is the wrapping function.

.. automodule:: src.algorithms.fmin_bbhhh
    :members: fmin_bhhh

""""""""""""""
Algorithm
""""""""""""""

This function performs the Quasi Newton-Rhapson algorithm with line search using the Hessian calculated
via the BHHH algorithm.

.. automodule:: src.algorithms.fmin_bbhhh
    :members: _minimize_bhhh

""""""""""""""
Hessian Calculation
""""""""""""""

In this function Hessian is calculated through the BHHH algorithm.

.. automodule:: src.algorithms.grad_hessian_vectorized
    :members: approx_hess_bhhh

""""""""""""""
Gradient Approximation
""""""""""""""

This function approximates the gradient of the objective funtion if it is not given as an input to *fmin_bbhhh*.

.. automodule:: src.algorithms.grad_hessian_vectorized
    :members: approx_fprime_ind
