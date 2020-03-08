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

Now we turn to have a look at how to implement bounds in our algorithms and how our algorithms perform in that scenario.
We set our bounds in the following way:::

    bounds = np.array([
    [-5, -5], # Lower Bound
    [5, 5]])  # Upper Bound

Our two algorithms and the L-BFGS-B give the following outcomes:::

    In[6]: fmin_bhhh(neg_log_binary_logistic, x0, data, bounds=bounds)
    Optimization terminated successfully.
         Current function value: 1456.998326
         Iterations: 10
    Out[6]:
    allvecs: [array([1., 1.]), array([0.87654655, 0.37912622]), array([0.81307231, 0.31523565]), array([0.78731086, 0.29517646]), array([0.79044908, 0.29459225]), array([0.79034258, 0.29454117]), array([0.79036428, 0.29454492]), array([0.79036157, 0.29454435]), array([0.79036193, 0.29454442]), array([0.79036189, 0.29454441]), array([0.79036189, 0.29454441])]
      fun: 1456.9983258198265
    hess_inv: array([[ 1508.07398218, -2540.34711358],
       [-2540.34711358, 24614.1895058 ]])
      jac: array([-1.51184944e-06, -3.77412718e-06])
    message: 'Optimization terminated successfully.'
      nit: 10
    status: 0
    success: True
        x: array([0.79036189, 0.29454441])

    In[7]: fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, bounds=bounds)
    Optimization terminated successfully.
             Current function value: 1456.998326
             Iterations: 20
    Out[7]:
     allvecs: [array([1., 1.]), array([1.09684122, 0.34619564]), array([1.08498527, 0.33167302]), array([0.83333005, 0.27279613]), array([0.82479022, 0.32141185]), array([0.82492236, 0.31403197]), array([0.82481991, 0.30905105]), array([0.82457451, 0.30567018]), array([0.82424252, 0.30334177]), array([0.82385313, 0.30169483]), array([0.82341565, 0.30049183]), array([0.82291411, 0.29956509]), array([0.82226688, 0.29876516]), array([0.82081561, 0.29766327]), array([0.79458805, 0.29322428]), array([0.79419733, 0.29493946]), array([0.79144468, 0.2943124 ]), array([0.79135294, 0.2946486 ]), array([0.79041434, 0.29454992]), array([0.79036183, 0.2945444 ]), array([0.79036189, 0.29454441])]
         fun: 1456.9983258198265
         jac: array([ 3.75485733e-08, -5.63228600e-08])
     message: 'Optimization terminated successfully.'
         nit: 20
      status: 0
     success: True
           x: array([0.79036189, 0.29454441])

     In[8]: opt.fmin_l_bfgs_b(
         sum_neg_log_bin_logistic, x0, approx_grad=1, bounds=[(-5, 5), (-5, 5)]
     )
     Out[8]:
     (array([0.79036008, 0.29454495]),
      1456.9983258281968,
      {'grad': array([-0.00404725,  0.01750777]),
       'task': b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
       'funcalls': 51,
       'nit': 11,
       'warnflag': 0})

Now we repeat this again with binding bounds which we choose in the following way:::

    bounds = np.array([[1, 0], [5, 5]])

Again we compare the performance of the three optimizers.::

    In[9]: fmin_bhhh(neg_log_binary_logistic, x0, data, bounds=bounds)
    Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 1457.518572
         Iterations: 4
    Out[9]:
    allvecs: [array([1., 1.]), array([1.        , 0.37912622]), array([1.        , 0.31730476]), array([1.        , 0.29798233]), array([1.        , 0.29571081])]
      fun: 1457.5185722940178
    hess_inv: array([[ 1609.038326  , -2786.49633572],
       [-2786.49633572, 25330.01697339]])
      jac: array([ 292.67819627, -503.71015451])
    message: 'Desired error not necessarily achieved due to precision loss.'
      nit: 4
    status: 2
    success: False
        x: array([1.        , 0.29571081])

    In[10]: fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, bounds=bounds)
    Warning: Desired error not necessarily achieved due to precision loss.
             Current function value: 1466.678892
             Iterations: 3
    Out[10]:
     allvecs: [array([1., 1.]), array([1.09684122, 0.34619564]), array([1.08498527, 0.33167302]), array([1.        , 0.27279613])]
         fun: 1466.6788923313547
         jac: array([  355.74543618, -1100.59348999])
     message: 'Desired error not necessarily achieved due to precision loss.'
         nit: 3
      status: 2
     success: False
           x: array([1.        , 0.27279613])

    In[12]: opt.fmin_l_bfgs_b(sum_neg_log_bin_logistic, x0, approx_grad=1, bounds=[(1, 5), (0, 5)])
    Out[12]:
    (array([1.        , 0.31735022]),
    1483.20994021217,
    {'grad': array([233.0335974,   0.       ]),
     'task': b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
     'funcalls': 24,
     'nit': 6,
     'warnflag': 0})
