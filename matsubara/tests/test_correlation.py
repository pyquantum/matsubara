"""
Tests for the correlation function calculations.
"""

import numpy as np
from numpy.testing import (
    run_module_suite,
    assert_,
    assert_array_almost_equal,
    assert_raises,
)
from matsubara.correlation import (
    sum_of_exponentials,
    biexp_fit,
    bath_correlation,
    underdamped_brownian,
    nonmatsubara_exponents,
    matsubara_exponents,
    matsubara_zero_analytical,
    coth,
    spectrum,
    spectrum_matsubara,
    spectrum_non_matsubara,
    _S,
    _A,
)


def test_sum_of_exponentials():
    """
    correlation: Test the sum of exponentials.
    """
    tlist = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Complex coefficients and frequencies.
    ck1 = [0.5 + 2j, -0.1]
    vk1 = [-0.5 + 3j, 1 - 1j]
    corr1 = sum_of_exponentials(ck1, vk1, tlist)

    y1 = np.array(
        [
            0.4 + 2.0j,
            -1.67084356 + 0.57764922j,
            -0.61828702 - 0.92938927j,
            0.84201641 + 0.0170242j,
            0.68968912 + 1.32694318j,
        ]
    )

    assert_array_almost_equal(corr1, y1)

    # Real coefficients and frequencies.
    ck2 = [0.5, -0.3]
    vk2 = [-0.9, -0.3]
    corr2 = sum_of_exponentials(ck2, vk2, tlist)

    y2 = np.array([0.2, 0.060602, -0.018961, -0.061668, -0.081994])

    assert_array_almost_equal(corr2, y2)


def test_biexp_fit():
    """
    correlation: Tests biexponential fitting.
    """
    tlist = np.linspace(0.0, 10, 100)
    ck = [-0.21, -0.13]
    vk = [-0.4, -1.5]

    corr = sum_of_exponentials(ck, vk, tlist)
    ck_fit, vk_fit = biexp_fit(tlist, corr)

    corr_fit = sum_of_exponentials(ck_fit, vk_fit, tlist)

    max_error = np.max(np.abs(corr - corr_fit))
    max_amplitude = np.max(np.abs(corr))

    assert max_error < max_amplitude / 1e3


def test_bath_correlation():
    """
    correlation: Tests for bath correlation function.
    """
    tlist = [0.0, 0.5, 1.0, 1.5, 2.0]
    lam, gamma, w0 = 0.4, 0.4, 1.0
    beta = np.inf
    w_cutoff = 10.0
    corr = bath_correlation(
        underdamped_brownian, tlist, [lam, gamma, w0], beta, w_cutoff
    )

    y = np.array(
        [
            0.07108,
            0.059188 - 0.03477j,
            0.033282 - 0.055529j,
            0.003295 - 0.060187j,
            -0.022843 - 0.050641j,
        ]
    )

    assert_array_almost_equal(corr, y)

    sd = np.arange(0, 10, 2)
    assert_raises(TypeError, bath_correlation, [sd, tlist, [0.1], beta, w_cutoff])


def test_exponents():
    """
    correlation: Tests the Matsubara and non Matsubara exponents.
    """
    lam, gamma, w0 = 0.2, 0.05, 1.0
    tlist = np.linspace(0, 100, 1000)

    # Finite temperature cases
    beta = 0.1
    N_exp = 200
    ck_nonmats = [0.190164 - 0.004997j, 0.21017 + 0.004997j]
    vk_nonmats = [-0.025 + 0.999687j, -0.025 - 0.999687j]
    ck1, vk1 = nonmatsubara_exponents(lam, gamma, w0, beta)

    assert_array_almost_equal(ck1, ck_nonmats)
    assert_array_almost_equal(vk1, vk_nonmats)

    ck2, vk2 = matsubara_exponents(lam, gamma, w0, beta, N_exp)
    corr_fit = sum_of_exponentials(
        np.concatenate([ck1, ck2]), np.concatenate([vk1, vk2]), tlist
    )

    corr = sum_of_exponentials(ck_nonmats, vk_nonmats, tlist)
    max_residue = np.max(np.abs(corr_fit - corr))
    max_amplitude = np.max(np.abs(corr))

    assert_(max_residue < max_amplitude / 1e5)

    # Lower temperature
    beta = 1.0
    N_exp = 100
    ck_nonmats = [0.011636 - 0.00046j, 0.031643 + 0.00046j]
    vk_nonmats = [-0.025 + 0.999687j, -0.025 - 0.999687j]
    ck1, vk1 = nonmatsubara_exponents(lam, gamma, w0, beta)

    assert_array_almost_equal(ck1, ck_nonmats)
    assert_array_almost_equal(vk1, vk_nonmats)

    ck2, vk2 = matsubara_exponents(lam, gamma, w0, beta, N_exp)
    corr_fit = sum_of_exponentials(
        np.concatenate([ck1, ck2]), np.concatenate([vk1, vk2]), tlist
    )

    corr = sum_of_exponentials(ck_nonmats, vk_nonmats, tlist)
    max_residue = np.max(np.abs(corr_fit - corr))
    max_amplitude = np.max(np.abs(corr))

    assert_(max_residue < max_amplitude / 1e3)

    # Zero temperature case
    beta = np.inf
    N_exp = 100
    ck_nonmats = [0.0, 0.020006]
    vk_nonmats = [-0.025 + 0.999687j, -0.025 - 0.999687j]
    ck1, vk1 = nonmatsubara_exponents(lam, gamma, w0, beta)

    assert_array_almost_equal(ck1, ck_nonmats)
    assert_array_almost_equal(vk1, vk_nonmats)

    ck2, vk2 = matsubara_exponents(lam, gamma, w0, beta, N_exp)
    mats_data_zero = matsubara_zero_analytical(lam, gamma, w0, tlist)
    ck_mats_zero = [-0.000208, -0.000107]
    vk_mats_zero = [-1.613416, -0.329604]
    ck20, vk20 = biexp_fit(tlist, mats_data_zero)

    assert_array_almost_equal(ck20, ck_mats_zero)
    assert_array_almost_equal(vk20, vk_mats_zero)


def test_spectrum():
    """
    correlation: Tests the Matsubara and non Matsubara spectrum.
    """
    bath_freq = 1.0
    bath_broad = 0.5
    coup_strength = 1

    # High temperature case
    beta = 0.01
    w = np.linspace(-5, 10, 200)
    s_matsu = spectrum_matsubara(w, coup_strength, bath_broad, bath_freq, beta)
    s_full = spectrum(w, coup_strength, bath_broad, bath_freq, beta)
    s_nonmatsu = spectrum_non_matsubara(w, coup_strength, bath_broad, bath_freq, beta)
    s_nonmatsu_neg = spectrum_non_matsubara(
        -w, coup_strength, bath_broad, bath_freq, beta
    )
    assert_array_almost_equal(s_full, s_nonmatsu)

    # Effective temperature
    div = np.divide(s_nonmatsu, s_nonmatsu_neg)
    log = np.log(div)
    effective_temperature = log / (w * beta)
    assert_array_almost_equal(effective_temperature, np.ones_like(w))
    # Low temperature case
    beta = 100
    w = np.linspace(-5, 10, 200)
    s_matsu = spectrum_matsubara(w, coup_strength, bath_broad, bath_freq, beta)
    s_full = spectrum(w, coup_strength, bath_broad, bath_freq, beta)
    s_nonmatsu = spectrum_non_matsubara(w, coup_strength, bath_broad, bath_freq, beta)
    residue = np.abs(s_full - s_nonmatsu)
    assert_(residue.all() != 0.0)

    # Effective temperature
    div = np.divide(s_nonmatsu, s_nonmatsu_neg)
    log = np.log(div)
    effective_temperature = log / (w * beta)
    residue = np.abs(effective_temperature - np.ones_like(w))
    assert_(residue.all() != 0.0)


if __name__ == "__main__":
    run_module_suite()
