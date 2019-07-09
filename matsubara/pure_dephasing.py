"""
Analytical computation of the dynamics for the pure dephasing case. 
"""
import numpy as np

from matsubara.correlation import underdamped_brownian, coth

from scipy.integrate import quad


def pure_dephasing_integrand(w, coup_strength, bath_broad, bath_freq, beta, t):
    """
    Calculates the pure dephasing integrand.
    
    Parameters
    ----------
    coup_strength: float
        The coupling strength parameter.

    bath_broad: float
        A parameter characterizing the FWHM of the spectral density.
        
    bath_freq: float
        The qubit frequency.
        
    beta: float
        The inverse temperature.
    
    t: float
        The time t.
    """
    fac = (1.0 - np.cos(w * t)) * (coth(w * beta / 2)) / w ** 2
    sd = -4 * underdamped_brownian(w, coup_strength, bath_broad, bath_freq)
    return sd * fac


def pure_dephasing_evolution(tlist, coup_strength, bath_broad, bath_freq, beta, w0):
    """
    Compute the pure dephasing evolution using the numerical integrand.
    """
    integrand = lambda t: quad(
        pure_dephasing_integrand,
        0.0,
        np.inf,
        args=(coup_strength, bath_broad, bath_freq, beta, t),
    )
    evolution = np.array([np.exp(1j * w0 * t + integrand(t)[0] / np.pi) for t in tlist])
    return evolution


def pure_dephasing_evolution_analytical(tlist, w0, ck, vk):
    """
    Computes the propagating function appearing in the pure dephasing model.
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    wq: float
        The qubit frequency in the Hamiltonian.

    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    evolution = np.array(
        [np.exp(-1j * w0 * t - correlation_integral(t, ck, vk)) for t in tlist]
    )
    return evolution


def correlation_integral(t, ck, vk):
    """
    Computes the integral sum function appearing in the pure dephasing model.
    
    If the correlation function is a sum of exponentials then this sum
    is given by:
    
    .. math::
        
        \int_0^{t}d\tau D(\tau) = \sum_k\frac{c_k}{\mu_k^2}e^{\mu_k t}
        + \frac{\bar c_k}{\bar \mu_k^2}e^{\bar \mu_k t}
        - \frac{\bar \mu_k c_k + \mu_k \bar c_k}{\mu_k \bar \mu_k} t
        + \frac{\bar \mu_k^2 c_k + \mu_k^2 \bar c_k}{\mu_k^2 \bar \mu_k^2}
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    t1 = np.sum(np.multiply(np.divide(ck, vk ** 2), np.exp(vk * t) - 1))
    t2 = np.sum(
        np.multiply(
            np.divide(np.conjugate(ck), np.conjugate(vk) ** 2),
            np.exp(np.conjugate(vk) * t) - 1,
        )
    )
    t3 = np.sum((np.divide(ck, vk) + np.divide(np.conjugate(ck), np.conjugate(vk))) * t)
    return 2 * (t1 + t2 - t3)
