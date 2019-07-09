"""
Tests for the HEOM class.
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
)
from qutip.operators import sigmaz, sigmax
from qutip import basis, expect
from qutip.solver import Options, Result, Stats

from matsubara.heom import HeomUB


def test_heom():
    """
    heom: Tests the HEOM method.
    """
    Q = sigmax()
    wq = 1.0
    lam, gamma, w0 = 0.2, 0.05, 1.0
    tlist = np.linspace(0, 200, 1000)
    Nc = 9
    # zero temperature case
    beta = np.inf

    Hsys = 0.5 * wq * sigmaz()
    initial_ket = basis(2, 1)
    rho0 = initial_ket * initial_ket.dag()
    omega = np.sqrt(w0 ** 2 - (gamma / 2.0) ** 2)
    a = omega + 1j * gamma / 2.0

    lam_coeff = lam ** 2 / (2 * (omega))

    options = Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)

    ck1, vk1 = nonmatsubara_exponents(lam, gamma, w0, beta)
    mats_data_zero = matsubara_zero_analytical(lam, gamma, w0, tlist)
    ck20, vk20 = biexp_fit(tlist, mats_data_zero)

    hsolver = HeomUB(
        Hsys,
        Q,
        lam_coeff,
        np.concatenate([ck1, ck20]),
        np.concatenate([-vk1, -vk20]),
        ncut=Nc,
    )

    output = hsolver.solve(rho0, tlist)
    result = np.real(expect(output.states, sigmaz()))

    # Ignore Matsubara
    hsolver2 = HeomUB(Hsys, Q, lam_coeff, ck1, -vk1, ncut=Nc)
    output2 = hsolver2.solve(rho0, tlist)
    result2 = np.real(expect(output2.states, sigmaz()))

    steady_state_error = np.abs(result[-1] - result2[-1])
    assert_(steady_state_error > 0.0)


if __name__ == "__main__":
    run_module_suite()
