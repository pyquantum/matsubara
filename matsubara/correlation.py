"""
Correlations for the underdamped Brownian motion spectral density.
"""


import numpy as np
from scipy.optimize import least_squares
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
    y = np.multiply(ck[0], np.exp(vk[0]*tlist))
    for p in range(1, len(ck)):
        y += np.multiply(ck[p], np.exp(vk[p]*tlist))
    return y


def biexp_fit(tlist, ydata,
              bounds=([0, -np.inf, 0, -np.inf], [10, 0, 10, 0]),
              loss='cauchy'):
    """
    Fits a bi-exponential function : ck[0] e^(-vk[0] t) + ck[1] e^(-vk[1] t)
    using `scipy.optimize.least_squares`. 

	Parameters
	----------
	tlist: array
		A list of time (x values).

	ydata: array
		The values for each time.

	bounds: array of arrays
		An array specifing the lower and upper bounds for each parameter.

	Returns
	-------
	ck, vk: array
		The array of coefficients and frequencies for the biexponential fit.
    """
    mindata = np.min(ydata)
    data = ydata/mindata
    fun = lambda x, t, y: np.power(x[0]*np.exp(x[1]*t) + (1-x[0])*np.exp(x[2]*t) - y, 2)
    x0 = [mindata/2, -.01, -.03]
    params = least_squares(fun, x0, bounds=bounds, 
    	loss=loss, args=(tlist, data))
    c1, v1, v2 = params.x
    ck = mindata*np.array([c1, (1-c1)])
    vk = np.array([v1, v2])
    return ck, vk


def underdamped_brownian(w, coup_strength, cav_broad, cav_freq):
    """
    Calculates the underdamped Brownian motion spectral density characterizing
    a bath of harmonic oscillators.

    Parameters
    ----------
    w: np.ndarray
        A 1D numpy array of frequencies.

    coup_strength: float
        The coupling strength parameter.

    cav_broad: float
        A parameter characterizing the FWHM of the spectral density, i.e.,
        the cavity broadening.

    cav_freq: float
        The cavity frequency.

    Returns
    -------
    spectral_density: ndarray
        The spectral density for specified parameters.
    """
    w0 = cav_freq
    lam = coup_strength
    gamma = cav_broad
    omega = np.sqrt(w0**2 - (gamma/2)**2)
    a = omega + 1j*gamma/2.
    aa = np.conjugate(a)
    prefactor = (lam**2)*gamma
    spectral_density = prefactor*(w/((w-a)*(w+a)*(w-aa)*(w+aa)))
    return spectral_density


def bath_correlation(spectral_density, tlist, params, beta, w_cut):
    """
    Calculates the bath correlation function (C) for a specific spectral
    density (J(w)) for an environment modelled as a bath of harmonic
    oscillators. If :math: `\beta` is the inverse temperature of the bath
    then the correlation is:

    :math:`C(t) = \frac{1}{\pi} \left[\int_{0}^{\infty} \coth
    (\beta \omega /2) \cos(\omega t) - i\sin(\omega t) \right]`

    where :math: `\beta = 1/kT` with T as the bath temperature and k as
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
        raise TypeError("""Spectral density should be a callable function
            f(w, args)""")

    corrR = []
    corrI = []

    coth = lambda x: 1/np.tanh(x)
    w_start = 0.

    integrandR = lambda w, t: np.real(spectral_density(w, *params) \
        *(coth(beta*(w/2)))*np.cos(w*t))
    integrandI = lambda w, t: np.real(-spectral_density(w, *params) \
        *np.sin(w*t))

    for i in tlist:
        corrR.append(np.real(quad(integrandR, w_start, w_cut, args=(i,))[0]))
        corrI.append(quad(integrandI, w_start, w_cut, args=(i,))[0])
    corr = (np.array(corrR) + 1j*np.array(corrI))/np.pi
    return corr
