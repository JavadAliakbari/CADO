import numpy as np
from tqdm import tqdm

from graph import Graph
from graph_atom import GraphAtom
from graph_atoms import GraphAtoms
from graph_func import GraphFunc
from graph_grad import GraphGrad
from perfect_graph import PerfectGraph
from utils import *


def find_next_atom(grad: GraphGrad) -> GraphAtom:
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
def update_e(grad: GraphGrad, atom: GraphAtom):
    tmp1 = np.einsum("ijk,jk->i", grad.grad_Rv, atom.R)
    # tmp1 = np.array([matrix_dot_product(matrix, atom.R) for matrix in grad.grad_Rv])
    # tmp2 = np.array([atom.b[label] for label in y])
    tmp2 = np.einsum("ij,j->i", grad.grad_pi, atom.b)
    tmp3 = grad.grad_L - config.mu1 * np.diag(tmp1) + config.mu2 * np.diag(tmp2)
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
    b = np.argmax(a, keepdims=True)

    return b


def calc_dot(grad: GraphGrad, atom: GraphAtom):
    term1 = matrix_dot_product(atom.e_term() * grad.grad_L)
    term2 = matrix_dot_product(atom.R_term() * grad.grad_Rv)
    term3 = matrix_dot_product(atom.b_term() * grad.grad_pi)

    return -term1 + config.mu1 * term2 - config.mu2 * term3


def Forward_step(atoms, f):
    sig = atoms.calc_sig()
    f.update_sig(sig)
    grad = f.calc_grad()

    new_atom = find_next_atom(grad)
    new_atom.update_coef(config.tau)
    atoms.add_atom(new_atom)
    # Dict.append(create_atom(eps, R, b))

    # Line 6 of CoGEnT
    gamma = 2 / (itr + 2)

    # Line 7 of CoGEnT(Line Search)
    atoms.update_coefficients(gamma)


def Enhancemnet_step(atoms: GraphAtoms, f: GraphFunc):
    for iter in range(config.GP_iters):
        grad = f.calc_grad()
        g = np.array([calc_dot(grad, atom) for atom in atoms.atoms])
        # g = A.T @ r  # Gradient of the function
        alpha = 1.0
        stop_flag = 0
        alpha_min = 1e-6
        alpha_red = 0.5
        c1 = 1e-3
        f_val = f.calc_objective()

        while (alpha > alpha_min) and (stop_flag == 0):
            # Projection onto the simplex
            # f = f.calc_objective()
            c = atoms.get_coefs()
            c_new = project_to_simplex(c - alpha * g, config.tau)

            # Check convergence tolerance
            tol_check = np.linalg.norm(c - c_new)
            if (alpha == 1) and (iter > 1):
                if tol_check <= config.GP_tol:
                    stop_flag = 1

            # Compute the new residual and function value
            f_new = f.calc_objective()
            # r_new = A @ c_new - y
            # f_new = 0.5 * (r_new.T @ r_new)

            # Check the Armijo condition (sufficient decrease condition)
            if f_new <= f_val + c1 * ((c_new - c).T @ g):
                break

            alpha *= alpha_red  # Reduce step size

        if stop_flag == 1:
            # Successful exit
            err = 0
            atoms.update_coefficients(c)
            return err

        if alpha < alpha_min:
            # Unsuccessful exit due to too small step length
            print(" *** No suitable step length found ***")
            err = 1
            return c, err

        # Update values for next iteration
        # r = r_new
        c = c_new
        atoms.update_coefficients(c)
        f_val = f_new

    # If maximum iterations reached
    err = 1
    atoms.update_coefficients(c)
    return err


if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    num_features = 5  # Dimension of the features at each node
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    graph = PerfectGraph(samples_per_cluster, num_features)

    ## Initialization
    atom0 = GraphAtom()
    atom0.initialize(n, num_features, c)
    atom0.update_coef(config.tau)

    atoms = GraphAtoms()
    atoms.add_atom(atom0)
    f = GraphFunc(graph=graph)

    # epsilon = np.random.randn(A_bar.shape[0])
    # R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    #     Rv_matrices.shape[1], rho_min, rho_max, noise_level=0.0, include_extremes=False
    # )
    # b = create_multiple_noisy_probability_vectors(1, num_classes, noise_level=100)[0]
    # Dict = [create_atom(epsilon, R, b)]
    # Coeffs = [tau]

    iter = 0
    for itr in tqdm(range(config.cogent_iters)):
        Forward_step(atoms, f)
        # Coeffs = update_coefficients_GP(Dict, Coeffs, tau)
        Enhancemnet_step(atoms, f)

        a = 0
