import numpy as np
import pytest
from grad_hessian_vectorized import approx_fprime_ind
from grad_hessian_vectorized import approx_hess_bhhh


@pytest.fixture
def setup_approx_fprime_ind():
    out = {}
    out["data"] = np.arange(-50, 50, 1)
    out["x0"] = np.array([4])

    def fun(b, data):
        return (5 - b * data) ** 2

    out["fun"] = fun
    return out


@pytest.fixture
def result_approx_fprime_ind():
    out = {}
    data = np.arange(-50, 50, 1)
    fprime = np.array(-2 * (5 - 4 * data) * data).reshape((100, 1))
    out["fprime"] = fprime.astype(float)
    out["approx_hessian"] = np.outer(fprime, fprime).diagonal().astype(float).sum()
    return out


def test_approx_fprime_ind(setup_approx_fprime_ind, result_approx_fprime_ind):
    grad = approx_fprime_ind(**setup_approx_fprime_ind)
    fprime = result_approx_fprime_ind["fprime"]
    assert np.allclose(grad, fprime)


def test_approx_hess_bhhh(result_approx_fprime_ind):
    approx_Hk = approx_hess_bhhh(result_approx_fprime_ind["fprime"])
    Hk = result_approx_fprime_ind["approx_hessian"]
    assert np.allclose(Hk, approx_Hk)


# Tests für bfgsrecb
