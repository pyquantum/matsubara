"""
This module provides a solver for the spin-boson model at zero temperature
using the hierarchy equations of motion (HEOM) method.
"""
# Authors: Shahnawaz Ahmed, Neill Lambert
# Contact: shahnawaz.ahmed95@gmail.com


import numpy as np

from copy import copy

from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from qutip import liouvillian, mat2vec, state_number_enumerate
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats

from scipy.misc import factorial
from scipy.sparse import lil_matrix
from scipy.integrate import ode


def add_at_idx(seq, k, val):
    """
    Add (subtract) a value in the tuple at position k
    """
    lst = list(seq)
    lst[k] += val
    return tuple(lst)


def prevhe(current_he, k, ncut):
    """
    Calculate the previous heirarchy index
    for the current index `n`.
    """
    nprev = add_at_idx(current_he, k, -1)
    if nprev[k] < 0:
        return False
    return nprev


def nexthe(current_he, k, ncut):
    """
    Calculate the next heirarchy index
    for the current index `n`.
    """
    nnext = add_at_idx(current_he, k, 1)
    if sum(nnext) > ncut:
        return False
    return nnext


def num_hierarchy(ncut, kcut):
    """
    Get the total number of auxiliary density matrices in the
    hierarchy.

    Parameters
    ==========
    ncut: int
        The Heirarchy cutoff

    kcut: int
        The cutoff in the correlation frequencies, i.e., how many
        total exponents are used.

    Returns
    =======
    num_hierarchy: int
        The total number of auxiliary density matrices in the hierarchy.
    """
    return int(factorial(ncut + kcut) / (factorial(ncut) * factorial(kcut)))


def _heom_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.
    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.
    excitations : integer
        The maximum numbers of dimension
    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1
    return nstates, state2idx, idx2state


def _heom_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.
    Example:
        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.
    state : list
        Current state in the iteration. Used internally.
    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.
    idx : integer
        Current index in the iteration. Used internally.
    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.
    """

    if state is None:
        state = np.zeros(len(dims))

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)

    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


class HeomUB(object):
    """
    The Heom class to tackle Heirarchy using the underdamped Brownian motion

    Parameters
    ----------
    hamiltonian: :class:`qutip.Qobj`
        The system Hamiltonian

    coupling: :class:`qutip.Qobj`
        The coupling operator

    coup_strength: float
        The coupling strength.

    ck: list
        The list of amplitudes in the expansion of the correlation function

    vk: list
        The list of frequencies in the expansion of the correlation function

    ncut: int
        The hierarchy cutoff.

	beta: float
		Inverse temperature, 1/kT. At zero temperature, beta is inf and we use
		an optimization for the non Matsubara terms.
    """

    def __init__(self, hamiltonian, coupling, coup_strength,
                 ck, vk, ncut, beta=np.inf):
        self.hamiltonian = hamiltonian
        self.coupling = coupling
        self.ck, self.vk = ck, vk
        self.ncut = ncut
        self.kcut = len(ck)
        nhe, he2idx, idx2he = _heom_state_dictionaries(
            [ncut + 1] * (len(ck)), ncut)

        self.nhe = nhe
        self.he2idx = he2idx
        self.idx2he = idx2he
        self.N = self.hamiltonian.shape[0]

        total_nhe = int(factorial(self.ncut + self.kcut) /
                        (factorial(self.ncut) * factorial(self.kcut)))
        self.total_nhe = total_nhe
        self.hshape = (total_nhe, self.N**2)
        self.L = liouvillian(self.hamiltonian, []).data
        self.grad_shape = (self.N**2, self.N**2)
        self.spreQ = spre(coupling).data
        self.spostQ = spost(coupling).data
        self.L_helems = lil_matrix(
            (total_nhe * self.N**2,
             total_nhe * self.N**2),
            dtype=np.complex)
        self.lam = coup_strength

    def populate(self, heidx_list):
        """
        Given a Hierarchy index list, populate the graph of next and
        previous elements
        """
        ncut = self.ncut
        kcut = self.kcut
        he2idx = self.he2idx
        idx2he = self.idx2he
        for heidx in heidx_list:
            for k in range(self.kcut):
                he_current = idx2he[heidx]
                he_next = nexthe(he_current, k, ncut)
                he_prev = prevhe(he_current, k, ncut)
                if he_next and (he_next not in he2idx):
                    he2idx[he_next] = self.nhe
                    idx2he[self.nhe] = he_next
                    self.nhe += 1

                if he_prev and (he_prev not in he2idx):
                    he2idx[he_prev] = self.nhe
                    idx2he[self.nhe] = he_prev
                    self.nhe += 1

    def grad_n(self, he_n):
        """
        Get the gradient term for the Hierarchy ADM at
        level n
        """
        c = self.ck
        nu = self.vk
        L = self.L.copy()
        gradient_sum = -np.sum(np.multiply(he_n, nu))
        sum_op = gradient_sum * np.eye(L.shape[0])
        L += sum_op

        # Fill in larger L
        nidx = self.he2idx[he_n]
        block = self.N**2
        pos = int(nidx * (block))
        self.L_helems[pos:pos + block, pos:pos + block] = L

    def grad_prev(self, he_n, k, prev_he):
        """
        Get prev gradient
        """
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        nk = he_n[k]
        norm_prev = nk

        # Non Matsubara terms
        if k == 0:
            norm_prev = np.sqrt(float(nk) / abs(self.lam))
            op1 = -1j * norm_prev * (-self.lam * spostQ)
        elif k == 1:
            norm_prev = np.sqrt(float(nk) / abs(self.lam))
            op1 = -1j * norm_prev * (self.lam * spreQ)
		# Matsubara terms
        else:
            norm_prev = np.sqrt(float(nk) / abs(c[k]))
            op1 = -1j * norm_prev * (c[k] * (spreQ - spostQ))

        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[prev_he]
        block = self.N**2
        rowpos = int(rowidx * (block))
        colpos = int(colidx * (block))
        self.L_helems[rowpos:rowpos + block, colpos:colpos + block] = op1

    def grad_next(self, he_n, k, next_he):
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ

        nk = he_n[k]

        # Non Matsubara terms
        if k < 2:
            norm_next = np.sqrt(self.lam * (nk + 1))
            op2 = -1j * norm_next * (spreQ - spostQ)
		# Non Matsubara terms
        else:
            norm_next = np.sqrt(abs(c[k]) * (nk + 1))
            op2 = -1j * norm_next * (spreQ - spostQ)

        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[next_he]
        block = self.N**2
        rowpos = int(rowidx * (block))
        colpos = int(colidx * (block))
        self.L_helems[rowpos:rowpos + block, colpos:colpos + block] = op2

    def rhs(self, progress=None):
        """
        Make the RHS
        """
        while self.nhe < self.total_nhe:
            heidxlist = copy(list(self.idx2he.keys()))
            self.populate(heidxlist)
        if progress is not None:
            bar = progress(total=self.nhe * self.kcut)

        for n in self.idx2he:
            he_n = self.idx2he[n]
            self.grad_n(he_n)
            for k in range(self.kcut):
                next_he = nexthe(he_n, k, self.ncut)
                prev_he = prevhe(he_n, k, self.ncut)
                if next_he and (next_he in self.he2idx):
                    self.grad_next(he_n, k, next_he)
                if prev_he and (prev_he in self.he2idx):
                    self.grad_prev(he_n, k, prev_he)

    def solve(self, rho0, tlist, options=None, progress=None, store_full=True):
        """
        Solve the Hierarchy equations of motion for the given initial
        density matrix and time.
        """
        if options is None:
            options = Options()

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        dt = np.diff(tlist)
        rho_he = np.zeros(self.hshape, dtype=np.complex)
        rho_he[0] = rho0.full().ravel("F")
        rho_he = rho_he.flatten()

        self.rhs()
        L_helems = self.L_helems.asformat("csr")
        r = ode(cy_ode_rhs)
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        r.set_initial_value(rho_he, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        if store_full:
            full_hierarchy = []
        if progress:
            bar = progress(total=n_tsteps - 1)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.reshape(self.hshape)
                r0 = r1[0].reshape(self.N, self.N).T
                output.states.append(Qobj(r0))

                if store_full:
                    r_heom = r.y.reshape(self.hshape)
                    full_hierarchy.append(r_heom)

                if progress:
                    bar.update()
        return output
