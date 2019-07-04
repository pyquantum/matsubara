"""
Calculate effective temperature from the spectrum of Matsubara and
non-matsubara correlations for the underdamped brownian motion density.
"""
import numpy as np
from matsubara.correlation import spectrum_matsubara, spectrum_non_matsubara, spectrum
import matplotlib.pyplot as plt


cav_freq = 1.
cav_broad = 0.5
coup_strength = 1

# High temperature case
beta = 0.001
w = np.linspace(-5, 10, 200)

matsu_spectrum = spectrum_matsubara(w, coup_strength, cav_broad, cav_freq, beta)
total_spectrum = spectrum(w, coup_strength, cav_broad, cav_freq, beta)

nonmatsu_spectrum = spectrum_non_matsubara(w, coup_strength, cav_broad, cav_freq, beta)
nonmatsu_spectrum_neg = spectrum_non_matsubara(-w, coup_strength, cav_broad, cav_freq, beta)

# Effective temperature
log = np.log(nonmatsu_spectrum/nonmatsu_spectrum_neg)
effective_beta = log/(w*beta)

plt.plot(w, total_spectrum, label = r"S(total)", color = "blue")
plt.plot(w, nonmatsu_spectrum, "--", label = r"$S_0(\omega)$", color = "orange")
plt.xlabel(r"$\omega$")
plt.ylabel(r"Spectrum")
plt.legend()
plt.show()

plt.figure(figsize=(5, 4))
plt.plot(w[w>0], effective_beta[w>0])
plt.ylim(-2, 2)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\beta_{eff}[\beta]$")
plt.show()


# Low temperature case
beta = 100
w = np.linspace(-5, 10, 200)

matsu_spectrum = spectrum_matsubara(w, coup_strength, cav_broad, cav_freq, beta)
total_spectrum = spectrum(w, coup_strength, cav_broad, cav_freq, beta)

nonmatsu_spectrum = spectrum_non_matsubara(w, coup_strength, cav_broad, cav_freq, beta)
nonmatsu_spectrum_neg = spectrum_non_matsubara(-w, coup_strength, cav_broad, cav_freq, beta)

# Effective temperature
log = np.log(nonmatsu_spectrum/nonmatsu_spectrum_neg)
effective_beta = log/(w*beta)

plt.plot(w, total_spectrum, label = r"$S(full)$", color = "blue")
plt.plot(w, nonmatsu_spectrum, "--", label = r"$S_0(\omega)$", color = "orange")
plt.xlabel(r"$\omega$")
plt.ylabel(r"Spectrum")
plt.legend()
plt.show()

plt.figure(figsize=(5, 4))
plt.plot(w[w>0], effective_beta[w>0])
plt.ylim(-.1, .1)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\beta_{eff}[\beta]$")
plt.show()
