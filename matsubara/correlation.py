"""
Correlations for the underdamped Brownian motion spectral density.
"""


import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import quad


def sum_of_exponentials(ck, vk, tlist):
    """
    Calculates the sum of exponentials for a set of `ck` and `vk` using
    `sum(ck[i]e^{vk[i]*t}`

    Parameters
    ----------
    ck: array
        An array of coefficients for the exponentials `ck`.

    vk: array
        An array of frequencies `vk`.

    tlist: array
        A list of times.

    Returns
    -------
    y: array
        A 1D array from a sum of exponentials.
    """
    tlist = np.array(tlist)
    y = np.multiply(ck[0], np.exp(vk[0] * tlist))
    for p in range(1, len(ck)):
        y += np.multiply(ck[p], np.exp(vk[p] * tlist))
    return y


def biexp_fit(
    tlist,
    ydata,
    ck_guess=[0.1, 0.5],
    vk_guess=[-0.5, -0.1],
    bounds=([0, -np.inf, 0, -np.inf], [np.inf, 0, np.inf, 0]),
    method="trf",
    loss="cauchy",
):
    """
    Fits a bi-exponential function : ck[0] e^(-vk[0] t) + ck[1] e^(-vk[1] t)
    using `scipy.optimize.least_squares`. 

    Parameters
    ----------
    tlist: array
        A list of time (x values).

    ydata: array
        The values for each time.

    guess: array
        The initial guess for the parameters [ck, vk]

    bounds: array of arrays
        An array specifing the lower and upper bounds for the parameters for
        the amplitude and the two frequencies.

    method, loss: str
        One of the `scipy.least_sqares` method and loss.

    Returns
    -------
    ck, vk: array
        The array of coefficients and frequencies for the biexponential fit.
    """
    mats_min = np.min(ydata)
    data = ydata / mats_min
    fun = lambda x, t, y: np.power(
        x[0] * np.exp(x[1] * t) + x[2] * np.exp(x[3] * t) - y, 2
    )
    x0 = [0.5, -1, 0.5, -1]
    # set the initial guess vector [ck1, ck2, vk1, vk2]
    params = least_squares(fun, x0, bounds=bounds, loss=loss, args=(tlist, data))
    c1, v1, c2, v2 = params.x
    ck = mats_min * np.array([c1, c2])
    vk = np.array([v1, v2])
    return ck, vk


def biexp_fit_constrained(
    tlist, ydata, w, lam, gamma, w0, method="trf", loss="cauchy", weight=1.0
):
    """
    Fits a bi-exponential function : ck[0] e^(-vk[0] t) + ck[1] e^(-vk[1] t)
    using `scipy.optimize.curve_fit` with an additional constraint 
    based on the fit giving positive power spectrum across a frequency range

    Parameters
    ----------
    tlist: array
        A list of time (x values).

    ydata: array
        The values for each time.
    
    w: linspace array of frequencise
    
    lam: coupling strength of the non-Matsubara term
    
    gamma: width of the non-Matsubara term
    
    w0: resonance of the non-Matsubara term

    guess: array
        The initial guess for the parameters [ck, vk]

    method, loss: str
        One of the `scipy.least_sqares` method and loss.
        
    weight: An optional weight for the cost function

    Returns
    -------
    ck, vk: array
        The array of coefficients and frequencies for the biexponential fit.
    """

    data = ydata

    ck_guess, vk_guess = biexp_fit(tlist, data)

    def St(w, lam, gamma, w0):
        Gam = gamma / 2.0
        Om = np.sqrt(w0 ** 2 - Gam ** 2)

        return (lam ** 2 / (2 * Om)) * 2 * (Gam) / ((w - Om) ** 2 + Gam ** 2)

    def cost(w, a1, a2, f1, f2, lam, gamma, w0):
        return (
            2 * (a1) * f1 / (w ** 2 + f1 ** 2)
            + 2 * (a2) * f2 / (w ** 2 + f2 ** 2)
            + St(w, lam, gamma, w0)
        )

    def fun(x, a1, a2, f1, f2):

        penal = 0.0
        for wt in w:
            if cost(wt, a1, a2, -f1, -f2, lam, gamma, w0) > 0.0:
                penalt = 0.0
            else:
                penalt = cost(wt, a1, a2, -f1, -f2, lam, gamma, w0)
            penal += penalt

        return a1 * np.exp(f1 * x) + a2 * np.exp(f2 * x) + abs(weight * penal)

    p0 = [ck_guess[0], ck_guess[1], vk_guess[0], vk_guess[1]]
    params, pcov = curve_fit(fun, tlist, np.real(data), method="trf", p0=p0)

    c1, c2, v1, v2 = params
    ck = np.array([c1, c2])
    vk = np.array([v1, v2])
    return ck, vk


def underdamped_brownian(w, coup_strength, bath_broad, bath_freq):
    """
    Calculates the underdamped Brownian motion spectral density characterizing
    a bath of harmonic oscillators.

    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.

    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    Returns
    -------
    spectral_density: ndarray
        The spectral density for specified parameters.
    """
    w0 = bath_freq
    lam = coup_strength
    gamma = bath_broad
    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)
    prefactor = (lam ** 2) * gamma
    spectral_density = prefactor * (w / ((w - a) * (w + a) * (w - aa) * (w + aa)))
    return spectral_density


def bath_correlation(spectral_density, tlist, params, beta, w_cut):
    r"""
    Calculates the bath correlation function (C) for a specific spectral
    density (J(w)) for an environment modelled as a bath of harmonic
    oscillators. If :math: `\beta` is the inverse temperature of the bath
    then the correlation is:

    .. math::

        C(t) = \frac{1}{\pi} \int_{0}^{\infty} 
        \coth(\beta \omega /2) \cos(\omega t) - i\sin(\omega t)

    where :math:`\beta = 1/kT` with T as the bath temperature and k as
    the Boltzmann's constant. If the temperature is zero, `beta` goes to
    infinity and we can replace the coth(x) term in the correlation
    function's real part with 1. At higher temperatures the coth(x)
    function behaves poorly at low frequencies.

    In general the intergration is for all values but since at higher
    frequencies, the spectral density is zero, we set a finite limit
    to the numerical integration.

    Assumptions:

        1. The bath is in a thermal state at a given temperature.
        2. The initial state of the environment is Gaussian.
        3. Bath operators are in a product state with the system intially.

    The `spectral_density` function is a callable, for example the Ohmic
    spectral density given as: `ohmic_sd = lambda w, eta: eta*w`

    Parameters
    ----------
    spectral_density: callable
        The spectral density for the given parameters.

    tlist : array
        A 1D array of times to calculate the correlation.

    params: ndarray
        A 1D array of parameters for the spectral density function.

    w_cut: float
        The cutoff value for the angular frequencies for integration.

    beta: float
        The inverse temperature of the bath.

    Returns
    -------
    corr: ndarray
        A 1D array giving the values of the correlation function for given
        time.
    """
    if not callable(spectral_density):
        raise TypeError(
            """Spectral density should be a callable function
            f(w, args)"""
        )

    corrR = []
    corrI = []

    coth = lambda x: 1 / np.tanh(x)
    w_start = 0.0

    integrandR = lambda w, t: np.real(
        spectral_density(w, *params) * (coth(beta * (w / 2))) * np.cos(w * t)
    )
    integrandI = lambda w, t: np.real(-spectral_density(w, *params) * np.sin(w * t))

    for i in tlist:
        corrR.append(np.real(quad(integrandR, w_start, w_cut, args=(i,))[0]))
        corrI.append(quad(integrandI, w_start, w_cut, args=(i,))[0])
    corr = (np.array(corrR) + 1j * np.array(corrI)) / np.pi
    return corr


def coth(x):
    """
    Calculates the coth function.

    Parameters
    ----------
    x: np.ndarray
        Any numpy array or list like input.

    Returns
    -------
    cothx: ndarray
        The coth function applied to the input.
    """
    return 1 / np.tanh(x)


def nonmatsubara_exponents(coup_strength, bath_broad, bath_freq, beta):
    """
    Get the exponentials for the correlation function for non-matsubara
    terms for the underdamped Brownian motion spectral density . (t>=0)

    Parameters
    ----------
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    beta: float
        The inverse temperature.

    Returns
    -------
    ck: ndarray
        A 1D array with the prefactors for the exponentials

    vk: ndarray
        A 1D array with the frequencies
    """
    w0 = bath_freq
    lam = coup_strength
    gamma = bath_broad

    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)
    coeff = lam ** 2 / (4 * omega)

    vk = np.array([1j * a, -1j * aa])

    if beta == np.inf:
        ck = np.array([0, 2.0])
    else:
        ck = np.array([coth(beta * (a / 2)) - 1, coth(beta * (aa / 2)) + 1])

    return coeff * ck, vk


def matsubara_exponents(coup_strength, bath_broad, bath_freq, beta, N_exp):
    """
    Calculates the exponentials for the correlation function for matsubara
    terms. (t>=0)

    Parameters
    ----------
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    beta: float
        The inverse temperature.

    N_exp: int
        The number of exponents to consider in the sum.

    Returns
    -------
    ck: ndarray
        A 1D array with the prefactors for the exponentials

    vk: ndarray
        A 1D array with the frequencies
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq
    N_exp = N_exp

    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)
    coeff = (-4 * gamma * lam ** 2 / np.pi) * ((np.pi / beta) ** 2)
    vk = np.array([-2 * np.pi * n / (beta) for n in range(1, N_exp)])
    ck = np.array(
        [
            n
            / (
                (a ** 2 + (2 * np.pi * n / beta) ** 2)
                * (aa ** 2 + (2 * np.pi * n / beta) ** 2)
            )
            for n in range(1, N_exp)
        ]
    )
    return coeff * ck, vk


def _matsubara_zero_integrand(t, coup_strength, bath_broad, bath_freq):
    """
    Integral for the zero temperature Matsubara exponentials.
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq

    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)

    prefactor = -(lam ** 2 * gamma) / np.pi
    integrand = lambda x: np.real(
        prefactor * ((x * np.exp(-x * t)) / ((a ** 2 + x ** 2) * (aa ** 2 + x ** 2)))
    )

    return quad(integrand, 0.0, np.inf)[0]


def matsubara_zero_analytical(coup_strength, bath_broad, bath_freq, tlist):
    """
    Calculates the analytical zero temperature value for Matsubara exponents.

    Parameters
    ----------
    tlist: array
        A 1D array of times to calculate the correlation function.

    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    Returns
    -------
    integrated: float
        The value of the integration at time "t".
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq

    return np.array(
        [_matsubara_zero_integrand(t, coup_strength, gamma, w0) for t in tlist]
    )


def _S(w, coup_strength, bath_broad, bath_freq, beta):
    """
    Calculates the symmetric part of the spectrum for underdamped brownian motion
    spectral density.
    
    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.
    
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    Returns
    -------
    integrated: float
        The value of the integration at time "t".
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq

    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)
    prefactor = -(lam ** 2) * gamma / (a ** 2 - aa ** 2)

    t1 = coth(beta * (a / 2)) * (a / (a ** 2 - w ** 2))
    t2 = coth(beta * (aa / 2)) * (aa / (aa ** 2 - w ** 2))
    return prefactor * (t1 - t2)


def _A(w, coup_strength, bath_broad, bath_freq, beta):
    """
    Calculates the anti-symmetric part of the spectrum for underdamped
    Brownian motion spectral density.
    
    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.
    
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    Returns
    -------
    integrated: float
        The value of the integration at time "t".
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq

    omega = np.sqrt(w0 ** 2 - (gamma / 2) ** 2)
    a = omega + 1j * gamma / 2.0
    aa = np.conjugate(a)
    prefactor = (lam ** 2) * gamma
    t1 = w / ((a ** 2 - w ** 2) * ((aa ** 2 - w ** 2)))
    return prefactor * t1


def spectrum_matsubara(w, coup_strength, bath_broad, bath_freq, beta):
    """
    Calculates the Matsubara part of the spectrum.
    
    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.
    
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    bath_freq: float
        The cavity frequency.

    Returns
    -------
    integrated: float
        The value of the integration at time "t".
    """
    lam = coup_strength
    gamma = bath_broad
    w0 = bath_freq
    return -_S(w, coup_strength, bath_broad, bath_freq, beta) + _A(
        w, coup_strength, bath_broad, bath_freq, beta
    ) * coth(beta * w / 2)


def spectrum_non_matsubara(w, coup_strength, bath_broad, bath_freq, beta):
    """
    Calculates the non Matsubara part of the spectrum.
    
    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.
    
    lam: float
        The coupling strength parameter.

    gamma: float
        A parameter characterizing the FWHM of the spectral density.
    
    beta: float
        Inverse temperature (1/kT) normalized to qubit frequency.
        deafult: inf
    """
    return _S(w, coup_strength, bath_broad, bath_freq, beta) + _A(
        w, coup_strength, bath_broad, bath_freq, beta
    )


def spectrum(w, coup_strength, bath_broad, bath_freq, beta):
    """
    Calculates the full spectrum for the spectral density.

    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.
    
    lam: float
        The coupling strength parameter.

    gamma: float
        A parameter characterizing the FWHM of the spectral density.
    
    beta: float
        Inverse temperature (1/kT) normalized to qubit frequency.
        deafult: inf
    """
    return spectrum_matsubara(
        w, coup_strength, bath_broad, bath_freq, beta
    ) + spectrum_non_matsubara(w, coup_strength, bath_broad, bath_freq, beta)
