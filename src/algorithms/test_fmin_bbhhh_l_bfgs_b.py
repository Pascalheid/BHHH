import numpy as np
import pytest
import scipy.stats as stats
from fmin_bbhhh import fmin_bhhh
from fmin_l_bfgs_b import fmin_l_bfgs_b


@pytest.fixture
def setup_data():

    out = {}
    np.random.seed(10)
    data = np.random.normal(5, 2, 10000)
    out["data"] = data

    def neg_log_dnorm(theta, data):

        return -np.log(stats.norm.pdf(data, theta[0], theta[1]))

    out["fun"] = neg_log_dnorm

    def sum_neg_log_dnorm(theta):
        return neg_log_dnorm(theta, data).sum()

    out["agg_fun"] = sum_neg_log_dnorm
    out["theta_ml"] = np.array([data.mean(), data.std()])
    out["theta_start"] = [0, 1]

    return out


def test_fmin_bhhh(setup_data):
    theta_est = fmin_bhhh(
        setup_data["fun"], setup_data["theta_start"], args=setup_data["data"]
    )["x"]
    assert np.allclose(theta_est, setup_data["theta_ml"])


def test_fmin_l_bfgs_b(setup_data):
    theta_est = fmin_l_bfgs_b(setup_data["agg_fun"], setup_data["theta_start"])["x"]
    assert np.allclose(theta_est, setup_data["theta_ml"])
