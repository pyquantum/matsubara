"""
Tests for the pure dephasing analytical computations.
"""

import numpy as np
from numpy.testing import (
    run_module_suite,
    assert_,
    assert_array_almost_equal,
    assert_raises,
)
from matsubara.pure_dephasing import (
    pure_dephasing_integrand,
    pure_dephasing_evolution,
    pure_dephasing_evolution_analytical,
)
from matsubara.correlation import (
    biexp_fit,
    nonmatsubara_exponents,
    matsubara_exponents,
    matsubara_zero_analytical,
)


def test_pure_dephasing():
    """
    pure_dephasing: Tests the pure_dephasing integrand.
    """
    coup_strength, bath_broad, bath_freq = 0.08, 0.4, 1.0

    # Test only at short times
    tlist = np.linspace(0, 20, 100)

    # Set qubit frequency to 0 to see only the dynamics due to the interaction.
    wq = 0
    # Zero temperature case
    beta = np.inf
    ck1, vk1 = nonmatsubara_exponents(coup_strength, bath_broad, bath_freq, beta)
    # ck2, vk2 = matsubara_exponents(lam, gamma, w0, beta, N_exp)

    mats_data_zero = matsubara_zero_analytical(
        coup_strength, bath_broad, bath_freq, tlist
    )
    ck20, vk20 = biexp_fit(tlist, mats_data_zero)

    ck = np.concatenate([ck1, ck20])
    vk = np.concatenate([vk1, vk20])

    pd_analytical = pure_dephasing_evolution(
        tlist, coup_strength, bath_broad, bath_freq, beta, wq
    )
    pd_numerical_fitting = pure_dephasing_evolution_analytical(tlist, wq, ck, vk)
    residue = np.abs(pd_analytical - pd_numerical_fitting)
    assert_(np.max(residue) < 1e-3)


if __name__ == "__main__":
    run_module_suite()
