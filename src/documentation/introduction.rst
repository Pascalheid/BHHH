.. _introduction:


************
Introduction
************

The purpose of this package is to implement the algorithm by Berndt-Hall-Hall-Hausman (BHHH) in Python
in order to numerically find the minimum of a function. On top of that, we further implement the Bounded Limited-Memory
Broyden–Fletcher–Goldfarb–Shanno alogorithm (L-BFGS-B).
This documentation starts off with a short description of those two algorithms.
In the next chapters we further report the docstrings of the functions used with some notes and conclude with some information on the practical
usage of our algorithms plus an example.

Both algorithms are part of the family of Quasi Newton-Rhapson methods. Those generally proceed in the following way.

    1. First the search direction :math:`p_k = -B_k^{-1} \nabla f(x_k)` needs to be found while :math:`p_k` is the direction, :math:`B_k` is the Hessian and :math:`\nabla f(x_k)` is the gradient of the objective function.

    2. Then :math:`\alpha_k` needs to be found by line search.

    3. And lastly :math:`x_k` is updated through :math:`x_{k+1} = x_k + \alpha_k p_k`

The two algorithms differ in the way :math:`B_k` is calculated.

.. _BHHH:

The BHHH
===============

The BHHH is only valid as long as the objective function f has the following form and if it is correctly specified:

.. math::

    f = \sum_{i=1}^{N} f_i(\beta_k)

It is therefore most often used to minimize negative log likelihood functions. This algorithm differs from other
Quasi Newton-Rhapson algorithms (such as the L-BFGS) in calculating the Hessian Matrix and hence in updating the
step size. For the BHHH case the Hessian matrix is calculated in the following way:

.. math::

    B_k = \sum_{i=1}^{N} \frac{\delta f_i(\beta_k)}{\delta \beta} \frac{\delta f_i(\beta_k)}{\delta \beta}^\prime

This is the outer product of the gradient of the individual funtions.
This calculation is performed in the function *approx_hess_bhhh*.
The actual algorithm for which the calculation of the Hessian is needed is done in the function *_minimize_bhhh*.
This function is wrapped by the function *fmin_bhhh*.

.. _L_BFGS:

The L-BFGS-B
===============

As previously mentioned we also implement the bounded, limited-memory BFGS algorithm. The general structure is fairly
similar although here the Hessian matrix is calculated differently and in comparison to BFGS in
a memory saving fashion (which is done in the function *bfgsrecb*).
The main algorithm can be found in the function *_minimize_lbfgsb* and it is wrapped by *fmin_l_bfgs_b*.

Note
===============

Both of our algorithms allow to include inequality constraints (bounds) which restricts the possible values among
the algorithms can look for a minimum.
Above that we for the line search we make us of the function *_line_search_wolfe12* which we take from scipy.
This line search algorithm satisifies the strong Wolfe conditions.
