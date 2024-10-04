import numpy as np

from graph_atom import GraphAtom
from graph_grad import GraphGrad
from utils import *


def find_next_atom(grad: GraphGrad) -> GraphAtom:
    """
    Alternating minimization to find the next atom.

    Parameters:
    - grad: grad of function f

    Returns:
    - A new atom
    """

    atom = GraphAtom()

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
def update_e(grad: GraphGrad, atom: GraphAtom):
    tmp1 = np.einsum("ijk,jk->i", grad.grad_Rv, atom.R)
    # tmp1 = np.array([matrix_dot_product(matrix, atom.R) for matrix in grad.grad_Rv])
    # tmp2 = np.array([atom.b[label] for label in y])
    tmp2 = np.einsum("ij,j->i", grad.grad_pi, atom.b)
    tmp3 = grad.grad_L - config.mu1 * np.diag(tmp1) + config.mu_2 * np.diag(tmp2)
    eigvecs, _, _ = np.linalg.svd(tmp3)
    epsilon = eigvecs[:, 0]

    return epsilon


## Step 2: Fix epsilon and b, update R
def update_R(grad: GraphGrad, atom: GraphAtom):
    # grad_R = np.sum(grad.grad_Rv * (atom.e**2)[:, np.newaxis, np.newaxis], axis=0)
    grad_R = np.einsum("ijk,i->jk", grad.grad_Rv, atom.e**2)
    R = atom.R
    for _ in range(config.iner_iter_GA_R):
        grad_step_R = R + config.iner_step_GA_R * grad_R
        R = project_to_set_S(grad_step_R, config.rho_min, config.rho_max)

    return R


## Step 3: Fix epsilon and R, update b (closed-form solution)
def update_b(grad: GraphGrad, atom: GraphAtom):
    a = np.einsum("i, ij-> j", atom.e**2, grad.grad_pi)
    b = np.argmax(a)

    return b


def calc_dot(grad: GraphGrad, atom: GraphAtom):
    term1 = matrix_dot_product(atom.e_term() * grad.grad_L)
    term2 = matrix_dot_product(atom.R_term() * grad.grad_Rv)
    term3 = matrix_dot_product(atom.b_term() * grad.grad_pi)

    return term1 - config.mu1 * term2 + config.mu2 * term3
