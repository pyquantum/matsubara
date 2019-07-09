############
Introduction
############

We provide the code to reproduce the results in :cite:`lambert2019virtual` using QuTiP.

The code is only for the zero temperature case where the correlation function can be expressed using four exponents - two non Matsubara contributions and a bi-exponential fitting of the infinite Matsubara sum. We will be extending it to a finite temperature case soon.

A special `matsubara.heom.HeomUB` class is provided to implement the Hierarchical Equations of Motion (HEOM) method adapted for the underdamped Brownian motion spectral density. We also implement the Reaction Coordinate (RC) and pseudomode method to tackle the same problem and show how a flat residual bath assumption (which is usually employed for the RC) gives the same results as a HEOM evolution without the Matsubara terms.

The results predict the emission of virtual photons from the ground state which is unphysical.
Adding the Matsubara terms with a biexponential fitting of the infinite sum recovers back the 
detailed balance in the system and gives the correct ground state. In the RC method, this amounts to assuming an Ohmic residual bath. Similarly, we can use three modes (one non Matsubara) and two Matsubara modes to recover the correct result using the pseudomodes although
this requires the evolution of a non-Hermitian Hamiltonian.

We also provide some comparision with the analytical solution of a pure dephasing model to quantify the error in the steady-state populations.

Lastly, we show how to calculate the virtual photon populations (bath occupations) from all the methods. While for the RC and pseudomode it is a straight-forward expectation calculation using the bath operators, in the HEOM formulation it is not so direct. In the HEOM method, we
recover the bath occupation from the Auxiliary Density Operators (ADOs) of the evolution.

The spin-boson Hamiltonian

.. math::

	H = \frac{\omega_q}{2}\sigma_z + \frac{\Delta}{2}  \sigma_x + \sum_k \omega_k b_k^{\dagger}b_k + \sigma_z \tilde{X}

where

.. math::
	 \tilde{X} = \sum_k \frac{g_k}{\sqrt{2\omega_k}} \left(b_k + b_k^{\dagger}\right)
	 
The underdamped Brownian motion spectral density

.. math::

	J(\omega) =\frac{ \gamma \lambda^2\omega}{(\omega^2-\omega_0^2)^2+\gamma^2 \omega^2}
