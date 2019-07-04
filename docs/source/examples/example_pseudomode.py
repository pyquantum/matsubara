"""
Calculate dynamics of the spin-boson model using Matsubara and
non Matsubara pseudomodes at zero temperature.
"""
import numpy as np
from matsubara.correlation import (sum_of_exponentials, biexp_fit,
                                   bath_correlation, underdamped_brownian,
                                   nonmatsubara_exponents, matsubara_exponents,
                                   matsubara_zero_analytical, coth)
from qutip.operators import sigmaz, sigmax
from qutip import (basis, expect, tensor, qeye, destroy, mesolve, 
                    spre, spost, liouvillian)
from qutip.solver import Options, Result, Stats

from matsubara.heom import HeomUB

import matplotlib.pyplot as plt

Q = sigmax()
wq = 1.
delta = 0.
beta = np.inf
coup_strength, bath_broad, bath_freq = 0.2, 0.05, 1.
tlist = np.linspace(0, 200, 1000)
Ncav = 4

mats_data_zero = matsubara_zero_analytical(coup_strength, bath_broad,
										   bath_freq, tlist)
# Fitting a biexponential function
ck20, vk20 = biexp_fit(tlist, mats_data_zero)

omega = np.sqrt(bath_freq**2 - (bath_broad/2.)**2)
lam_renorm = coup_strength**2/(2*omega)

lam2 = np.sqrt(lam_renorm)

# Construct the pseudomode operators with one extra underdamped pseudomode
sx = tensor(sigmax(), qeye(Ncav))
sm = tensor(destroy(2).dag(), qeye(Ncav))
sz = tensor(sigmaz(), qeye(Ncav))
a = tensor(qeye(2), destroy (Ncav))

Hsys = 0.5*wq*sz + 0.5*delta*sx + omega*a.dag()*a + lam2*sx*(a + a.dag())
initial_ket = basis(2, 1)
psi0=tensor(initial_ket, basis(Ncav, 0))

options = Options(nsteps=1500, store_states=True, atol=1e-13, rtol=1e-13)
c_ops = [np.sqrt(bath_broad)*a]
e_ops = [sz, ]
pseudomode_no_mats = mesolve(Hsys, psi0, tlist, c_ops, e_ops, options=options)
output = (pseudomode_no_mats.expect[0] + 1)/2

# Construct the pseudomode operators with three extra pseudomodes
# One of the added modes is the underdamped pseudomode and the two extra are
# the matsubara modes.
sx = tensor(sigmax(), qeye(Ncav), qeye(Ncav), qeye(Ncav))
sm = tensor(destroy(2).dag(), qeye(Ncav), qeye(Ncav), qeye(Ncav))
sz = tensor(sigmaz(), qeye(Ncav), qeye(Ncav), qeye(Ncav))
a = tensor(qeye(2), destroy(Ncav), qeye(Ncav), qeye(Ncav))

b = tensor(qeye(2), qeye(Ncav), destroy(Ncav), qeye(Ncav))
c = tensor(qeye(2), qeye(Ncav), qeye(Ncav), destroy(Ncav))

lam3 =1.0j*np.sqrt(-ck20[0])
lam4 =1.0j*np.sqrt(-ck20[1])

Hsys = 0.5*wq*sz + 0.5*delta*sx + omega*a.dag()*a + lam2*sx*(a + a.dag())
Hsys = Hsys + lam3*sx*(b+b.dag())
Hsys = Hsys + lam4*sx*(c + c.dag())

psi0 = tensor(initial_ket, basis(Ncav,0), basis(Ncav,0), basis(Ncav,0))
c_ops = [np.sqrt(bath_broad)*a, np.sqrt(-2*vk20[0])*b, np.sqrt(-2*vk20[1])*c]
e_ops = [sz, ]
L = -1.0j*(spre(Hsys)-spost(Hsys)) + liouvillian(0*Hsys,c_ops)
pseudomode_with_mats = mesolve(L, psi0, tlist, [], e_ops, options=options)
output2 = (pseudomode_with_mats.expect[0] + 1)/2

plt.plot(tlist, output, color="b", label="Psuedomode (no Matsubara)")
plt.plot(tlist, output2, color="r", label="Psuedomode (Matsubara)")
plt.xlabel("t ($1/\omega_0$)")
plt.ylabel(r"$\langle 1 | \rho | 1 \rangle$")
plt.legend()
plt.savefig("pseudo.png", bbox_inches="tight")
plt.show()
