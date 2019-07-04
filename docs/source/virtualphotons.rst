#########################
Virtual photon population
#########################


In the ultrastrong coupling regime where the qubit-environment coupling is comparable to the bath frequencies, the hybridized system-environment “groundstate” (which in principle should be the steady-state at zero temperature) contains a finite population of photons which cannot be directly observed. These "virtual photon" populations can be extracted from the methods discussed here. The Matsubara terms are crucial to get the correct photon population in a single collective mode, and make sure that the population is trapped in the ground state.

In case of the pseudomode and the Reaction Coordinate method, calculating the virtual photon population is a straight-forward calculation of the expectation of the bath operators. These are approximately the same as the dominant modes from Eq (19) in :cite:`lambert2019virtual` which can be supplied to `mesolve` as discussed in the previous examples:

.. code-block:: python

	# Three pseudomodes for the non Matsubara and Matsubara modes
	a = tensor(qeye(2), destroy(Ncav), qeye(Ncav), qeye(Ncav))
	e_ops = [sz, a.dag()*a]
	pseudomode_with_mats = mesolve(Hsys, psi0, tlist, c_ops, e_ops, options=options)
	virtual_population = pseudomode_with_mats.expect[1]


But this is not so obvious in the HEOM method. Here, the Auxiliary Density Operators (ADOs) of the evolution contain the information about the bath operators (:cite:`zhu2012explicit`, :cite:`song2017hierarchical`) and we have calculated them in Eq (20) of :cite:`lambert2019virtual`. Here we show how to extract them from the full HEOM evolution.

