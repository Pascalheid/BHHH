.. _example:

**********
Usage and Example
**********

In this section we run through the example of minimizing the negative log likelihood of a binary logistic model presenting how to
utilize our two algorithms and present their performance.
Let us first set up simulated data for our example and import all necessary packages::

    import numpy as np
    import scipy.optimize as opt
    import scipy.stats as stats
    import statsmodels.api as sm
    from fmin_bbhhh import fmin_bhhh
    from fmin_l_bfgs_b import fmin_l_bfgs_b

    # Generate data
    np.random.seed(100)
    X = np.random.randn(10000, 2) * np.random.uniform(0.5, 4, (1, 2)) + np.random.uniform(-20, 20, (1, 2))
    b_true = np.random.randn(2, 1)
    Z = X.dot(b_true)
    Pr = 1 / (1 + np.exp(- Z))
    y = np.random.binomial(1, Pr)
    data = np.hstack((y, X))

After having simulated the data we observe that the true parameters for :math:`\beta` are:::

    In [1]: b_true
    Out[1]:
    array([[0.82506968],
       [0.30193531]])

Let us now define the function that we need to minimize in order to use our BHHH algorithm and let us add the starting value
:math:`x_0` (fixed for the rest of our examples).::

    def neg_log_binary_logistic(theta, data):

        return - (data[:, 0] * data[:, 1:].dot(theta) - np.log(1 + np.exp(data[:, 1:].dot(theta))))

    # Starting point (1, 1)
    x0 = np.ones(2)

It is crucial to note that for the BHHH we require a function that gives out the negative individual contributions to the overall sum
but not the sum of those contributions. With this set up we can now invoke our BHHH algorithm first without any bounds.::

    In[2]: fmin_bhhh(neg_log_binary_logistic, x0, data)
    Optimization terminated successfully.
         Current function value: 1456.998326
         Iterations: 9
    Out[2]:
    allvecs: [array([1., 1.]), array([0.87654655, 0.37912622]), array([0.81307231, 0.31523565]), array([0.78731086, 0.29517646]), array([0.79044908, 0.29459225]), array([0.79034258, 0.29454117]), array([0.79036428, 0.29454492]), array([0.79036157, 0.29454435]), array([0.79036193, 0.29454442]), array([0.79036189, 0.29454441])]
      fun: 1456.9983258198265
    hess_inv: array([[ 1508.0739846 , -2540.34711689],
       [-2540.34711689, 24614.18951563]])
      jac: array([-6.38650487e-06, -1.59479073e-05])
    message: 'Optimization terminated successfully.'
      nit: 9
    status: 0
    success: True
        x: array([0.79036189, 0.29454441])

Let us compare this result to our L-BFGS-B algorithm first. This algorithm takes the negative sum of the individual contributions
as an input. Hence, we first of define our negative log likelihood function:::

    def sum_neg_log_bin_logistic(x0):

        return neg_log_binary_logistic(x0, data).sum()

Our L-BFGS-B gives the following outcome:::

    In[3]: fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0)
    ns reached maximum size
    Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 1456.998326
         Iterations: 8
    Out[3]:
    allvecs: [array([1., 1.]), array([1.09684122, 0.34619564]), array([0.78879215, 0.27399497]), array([0.76336012, 0.29006797]), array([0.80737619, 0.2970738 ]), array([0.79098248, 0.29466063]), array([0.79034598, 0.29454262]), array([0.79035222, 0.29454521]), array([0.79036189, 0.29454442])]
     fun: 1456.9983258198265
     jac: array([-1.27852892e-05,  7.53599866e-05])
    message: 'Desired error not necessarily achieved due to precision loss.'
     nit: 8
    status: 2
    success: False
       x: array([0.79036189, 0.29454442])

Now we compare this result to that of the binary logit regression from statsmodels:::

    In[4]: sm.Logit(data[:, 0], data[:, 1:]).fit().summary()
    Optimization terminated successfully.
         Current function value: 0.145700
         Iterations 9
    Out[4]:
    <class 'statsmodels.iolib.summary.Summary'>
    """
                           Logit Regression Results
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                10000
    Model:                          Logit   Df Residuals:                     9998
    Method:                           MLE   Df Model:                            1
    Date:                Sun, 08 Mar 2020   Pseudo R-squ.:                  0.3627
    Time:                        23:30:46   Log-Likelihood:                -1457.0
    converged:                       True   LL-Null:                       -2286.2
                                        LLR p-value:                     0.000
    ==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.7904      0.027     29.309      0.000       0.738       0.843
    x2             0.2945      0.007     43.234      0.000       0.281       0.308
    ==============================================================================
    """

Lastly, we check the performance of the L-BFGS-B by scipy:::

    In[5]: opt.fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, approx_grad = 1)
    Out[5]:
    (array([0.79036224, 0.29454454]),
    1456.9983258200318,
    {'grad': array([0.00029559, 0.0026148 ]),
    'task': b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
    'funcalls': 39,
    'nit': 9,
    'warnflag': 0})



This is a sample::

    random.seed(123)
    X = np.random.randn(10000, 2) * np.random.uniform(0.5, 4, (1, 2)) + np.random.uniform(-20, 20, (1, 2))

    b_true = np.random.randn(2, 1)
    Z = X.dot(b_true)

    Pr = 1 / (1 + np.exp(-Z))
    y = np.random.binomial(1, Pr)
    data = np.hstack((y, X))

    def neg_log_binary_logistic(theta, data):

    return -(data[:, 0] * data[:, 1:].dot(theta) - np.log(1 + np.exp(data[:, 1:].dot(theta))))

That is enough.

Some Output::

    In [5]: y
    Out[5]:
    array([[0],
           [0],
           [0],
           ...,
           [0],
           [0],
           [0]])
