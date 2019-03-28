"""
Tests for the correlation function calculations.
"""

import numpy as np
from numpy.testing import (run_module_suite, assert_,
                           assert_array_almost_equal, assert_raises)
from matsubara.correlation import (sum_of_exponentials, biexp_fit,
                                   bath_correlation, underdamped_brownian)


def test_sum_of_exponentials():
    """
    correlation: Test the sum of exponentials.
    """
    tlist = [0., 0.5, 1., 1.5, 2.]

    # Complex coefficients and frequencies.
    ck1 = [0.5 + 2j, -0.1]
    vk1 = [-0.5 + 3j, 1 - 1j]
    corr1 = sum_of_exponentials(ck1, vk1, tlist)

    y1 = np.array([0.4 + 2.j, -1.67084356 + 0.57764922j,
                  -0.61828702 - 0.92938927j, 
                  0.84201641 + 0.0170242j,
                  0.68968912 + 1.32694318j])

    assert_array_almost_equal(corr1, y1)

    # Real coefficients and frequencies.
    ck2 = [0.5, -0.3]
    vk2 = [-.9, -.3]
    corr2 = sum_of_exponentials(ck2, vk2, tlist)

    y2 = np.array([0.2, 0.060602, -0.018961, -0.061668,
                   -0.081994])

    assert_array_almost_equal(corr2, y2)


def test_biexp_fit():
    """
    correlation: Tests biexponential fitting.
    """
    tlist = np.linspace(0., 10, 100)
    ck = [-0.21, -0.13]
    vk = [-0.4, -1.5]

    corr = sum_of_exponentials(ck, vk, tlist)
    ck_fit, vk_fit = biexp_fit(tlist, corr)

    corr_fit = sum_of_exponentials(ck_fit, vk_fit, tlist)

    max_error = np.max(np.abs(corr - corr_fit))
    max_amplitude = np.max(np.abs(corr))

    assert(max_error < max_amplitude/1e3)


def test_bath_correlation():
    """
    correlation: Tests for bath correlation function.
    """
    tlist = [0., 0.5, 1., 1.5, 2.]
    lam, gamma, w0 = 0.4, 0.4, 1.
    beta = np.inf
    w_cutoff = 10.
    corr = bath_correlation(underdamped_brownian, tlist,
                            [lam, gamma, w0], beta, w_cutoff)

    y = np.array([0.07108, 0.059188-0.03477j, 0.033282-0.055529j,
                  0.003295-0.060187j, -0.022843-0.050641j])

    assert_array_almost_equal(corr, y)

    sd = np.arange(0, 10, 2)
    assert_raises(TypeError, bath_correlation, [sd, tlist, [0.1], beta,
                                                w_cutoff])


if __name__ == "__main__":
    run_module_suite()
