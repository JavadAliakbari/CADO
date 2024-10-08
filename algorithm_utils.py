import numpy as np

from coefs import Coefs
from graph_atom import GraphAtom
from atoms import Atoms
from graph_func import GraphFunc
from graph_sig import GraphSig
from utils import *


def find_next_atom(grad: GraphSig) -> GraphAtom:
    """
    Alternating minimization to find the next atom.

    Parameters:
    - grad: grad of function f

    Returns:
    - A new atom
    """
    n, d, c = grad.get_params()

    atom = GraphAtom()
    atom.initialize(n, d, c)

    # Alternating minimization
    for it in range(config.iter_or_cycles_to_find_next_atom):
        epsilon = update_e(grad, atom)
        atom.update_e(epsilon)

        R = update_R(grad, atom)
        atom.update_R(R)

        b = update_b(grad, atom)
        atom.update_b(b)

    # Return the next atom in the form [epsilon @ epsilon^T, R, b]
    return atom


## Step 1: Fix R and b, update epsilon
def update_e(grad: GraphSig, atom: GraphAtom):
    tmp1 = np.einsum("ijk,jk->i", grad.Rv, atom.R)
    # tmp1 = np.array([matrix_dot_product(matrix, atom.R) for matrix in grad.grad_Rv])
    # tmp2 = np.array([atom.b[label] for label in y])
    tmp2 = np.einsum("ij,j->i", grad.pi, atom.b)
    tmp3 = grad.L - config.mu1 * np.diag(tmp1) + config.mu2 * np.diag(tmp2)
    # S, eigvecs = np.linalg.eigh(tmp3)
    eigvecs, _, _ = np.linalg.svd(tmp3)
    epsilon = eigvecs[:, 0]

    return epsilon


## Step 2: Fix epsilon and b, update R
def update_R(grad: GraphSig, atom: GraphAtom):
    # grad_R = np.sum(grad.grad_Rv * (atom.e**2)[:, np.newaxis, np.newaxis], axis=0)
    grad_R = np.einsum("ijk,i->jk", grad.Rv, atom.e**2)
    R = atom.R
    for _ in range(config.iner_iter_GA_R):
        grad_step_R = R + config.iner_step_GA_R * grad_R
        R = project_to_set_S(grad_step_R, config.rho_min, config.rho_max)

    return R


## Step 3: Fix epsilon and R, update b (closed-form solution)
def update_b(grad: GraphSig, atom: GraphAtom):
    a = np.einsum("i, ij-> j", atom.e**2, grad.pi)
    b = np.zeros(grad.pi.shape[-1], dtype=float)
    b[np.argmax(a)] = 1

    return b
