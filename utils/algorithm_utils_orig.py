from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Atom.sig import Sig
from Graph_Atom.graph_atom import GraphAtom
from Graph_Atom.graph_sig import EmptySig, GraphSig
from Graph_Atom.graph_func import GraphFunc
from utils.graph import Graph
from utils.utils import *


def find_next_atom(grad: GraphSig, itr) -> GraphAtom:
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
    # atom.R = None

    max_iter = (
        config.iter_or_cycles_to_find_next_atom
        if config.mu1 != 0 or config.mu2 != 0
        else 1
    )
    th_old = np.inf

    for it in range(max_iter):
        epsilon, λ = update_e(grad, atom, it, itr)
        atom.update_e(epsilon)

        if config.mu1 != 0:
            R = update_R(grad, atom, it, itr)
            atom.update_R(R)

        if config.mu2 != 0:
            b = update_b(grad, atom)
            atom.update_b(b)

        th = grad.dot(atom.calc_sig_term())
        if abs(th_old - th) < 1e-6:
            break
        th_old = th

    return atom, λ


## Step 1: Fix R and b, update epsilon
def update_e(grad: GraphSig, atom: GraphAtom, iter=0, itr=0):
    scale_factor = (config.rho_max / config.rho_min) ** 2
    if atom.R is not None:
        tmp1 = np.einsum("ijk,jk->i", grad.Rv, atom.R)
    else:
        tmp1 = 0

    tmp1 /= scale_factor
    tmp2 = np.einsum("ij,j->i", grad.pi, atom.b)
    tmp3 = grad.L + np.diag(tmp1 + tmp2)
    # S, eigvecs = np.linalg.eigh(-tmp3)
    N = tmp3.shape[0]
    try:
        λ, evecs_large = sc.linalg.eigh(-tmp3, subset_by_index=(N - 1, N - 1))
        epsilon = evecs_large[:, 0]
    except:
        λ, evecs_large = np.linalg.eigh(-tmp3)
        epsilon = evecs_large[:, -1]
        λ = λ[-1]

    if plot:
        sub_folder = f"{save_path}{itr}/e/"
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        plt.figure()
        plt.plot(epsilon, "-*")
        plt.savefig(f"{sub_folder}{iter}.png")
        plt.close()

    return epsilon, λ


## Step 2: Fix epsilon and b, update R
def update_R(grad: GraphSig, atom: GraphAtom, iter, itr=0):
    grad_R = np.einsum("ijk,i->jk", grad.Rv, atom.e**2)

    S, U = np.linalg.eigh(grad_R)

    if plot:
        sub_folder = f"{save_path}{itr}/R/"
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        plt.figure()
        plt.bar(np.arange(S.shape[0]), S)
        plt.savefig(f"{sub_folder}{iter}.png")
        plt.close()

    mask1 = S < 0
    S[mask1] = config.rho_max - config.rho_min
    S[~mask1] = 0
    R = U @ np.diag(S) @ U.T

    return R


## Step 2: Fix epsilon and b, update R
def update_R2(grad: GraphSig, atom: GraphAtom, iter, itr=0):
    grad_R = np.einsum("ijk,i->jk", grad, atom.e**2)

    S, U = np.linalg.eigh(grad_R)

    if plot:
        sub_folder = f"{save_path}{itr}/R/"
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        plt.figure()
        plt.bar(np.arange(S.shape[0]), S)
        plt.savefig(f"{sub_folder}{iter}.png")
        plt.close()

    mask1 = S > config.rho_max
    S[mask1] = config.rho_max
    mask2 = S < config.rho_min
    S[mask2] = config.rho_min
    R = U @ np.diag(S) @ U.T

    return R


## Step 3: Fix epsilon and R, update b (closed-form solution)
def update_b(grad: GraphSig, atom: GraphAtom):
    a = np.einsum("i, ij-> j", atom.e**2, grad.pi)
    b = np.zeros(grad.pi.shape[-1], dtype=float)
    b[np.argmin(a)] = 1

    return b


def Gamma_line_search(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    """
    Perform a line search to find the optimal gamma value that minimizes the function f.

    This function uses a binary search approach to find the gamma value in the range [0, 1]
    that minimizes the function f when applied to the signal calculated from the scaled coefficients.

    Parameters:
    atoms (Atoms): An object that contains the method `calc_sig` to calculate the signal from coefficients.
    f (GraphFunc): A function that takes a signal and returns a scalar value to be minimized.
    coefs (Coefs): An object that contains the method `scale_coefs` to scale the coefficients by a given gamma value.

    Returns:
    float: The gamma value in the range [0, 1] that minimizes the function f.
    """
    gamma_min, gamma_max = 0, 1

    for _ in range(10):
        gamma_mean = (gamma_max + gamma_min) / 2
        coefs_gamma_mean = Coefs.scale_coefs(coefs, gamma_mean)
        sig_gamma_mean = atoms.calc_sig(coefs_gamma_mean)
        f_gamma_mean = f(sig_gamma_mean)

        if f_gamma_mean < f(atoms.calc_sig(Coefs.scale_coefs(coefs, gamma_min))):
            gamma_max = gamma_mean
        else:
            gamma_min = gamma_mean

    return gamma_min


def remove_atom(atoms: Atoms, f: GraphFunc, coefs: Coefs) -> int:
    """
    Identifies the index of the atom whose removal minimizes the value of the given function `f`
    when applied to the residual signature.

    Parameters:
    atoms (Atoms): The collection of atoms.
    f (GraphFunc): A function that evaluates the residual signature.
    coefs (Coefs): The coefficients associated with the atoms.

    Returns:
    int: The index of the atom whose removal minimizes the value of `f`. Returns 0 if there is only one coefficient.
    """
    if len(coefs) == 1:
        return 0
    # zxzjm
    # sig = atoms.calc_sig(coefs)

    min_val = np.inf
    arg_min_idx = -1
    for idx in range(len(coefs)):
        new_coefs = deepcopy(coefs)
        new_atoms = deepcopy(atoms)
        new_coefs.remove_coef(idx)
        # new_atoms.remove_atom(atoms[idx])
        new_atoms.remove_idx(idx)
        c_new = project_to_simplex(new_coefs, config.tau)
        new_coefs = Coefs(c_new)
        if config.enhance:
            new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
        residual_sig = new_atoms.calc_sig(new_coefs)
        # single_sig = atoms[idx].calc_sig_term()
        # single_sig = coefs[idx] * single_sig
        # residual_sig = sig - single_sig
        val = f(residual_sig)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx


def Forward_step(atoms: Atoms, f: GraphFunc, coefs: Coefs, sig, itr):
    """
    Perform a forward step in the CoGEnT algorithm.

    Parameters:
    atoms (Atoms): The current set of atoms.
    f (GraphFunc): The graph function used to calculate the gradient.
    coefs (Coefs): The coefficients associated with the atoms.
    sig: The signal or data used for gradient calculation.
    itr (int): The current iteration number.

    Returns:
    Coefs: The updated coefficients after the forward step.
    """
    grad = f.calc_grad(sig)

    new_atom, λ = find_next_atom(grad, itr)

    atoms.add_atom(new_atom)

    # coefs.add_coef(λ)
    # c = project_to_simplex(coefs.get_coefs(), config.tau)
    # coefs.update_coefficients(c)

    coefs.add_coef(config.tau)
    # Line 6 of CoGEnT
    if config.line_search:
        gamma = Gamma_line_search(atoms, f, coefs)
    else:
        gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return atoms, coefs


def Enhancemnet_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    """
    Perform the enhancement step for the given coefficients.

    Parameters:
    atoms (Atoms): The atoms object containing the necessary methods for calculations.
    f (GraphFunc): The graph function object used to calculate gradients and function values.
    coefs (Coefs): The coefficients object to be updated.

    Returns:
    Coefs: The updated coefficients object.

    Description:
    This function performs an enhancement step to update the coefficients using gradient projection.
    It iteratively adjusts the coefficients to minimize the function value while ensuring they lie
    within a simplex defined by a given tau value. The function uses a gradient projection method
    with backtracking line search to find the optimal step size. The process stops when the
    convergence tolerance is met or the step size becomes too small.
    """
    if len(coefs) <= 1:
        coefs.update_coefficients([config.tau])
        return coefs

    alpha_min, alpha_red, c1 = 0.1, 0.5, 1e-3
    sig = atoms.calc_sig(coefs)
    f_val = f(sig)
    grad = f.calc_grad(sig)

    for _ in range(config.GP_iters):
        g = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms])
        # g = g / np.linalg.norm(g)
        alpha, stop_flag = sig.n, False

        while alpha > alpha_min and not stop_flag:
            c_new = project_to_simplex(coefs.get_coefs() - alpha * g, config.tau)
            grad_diff = (c_new - coefs.get_coefs()).T @ g
            if grad_diff >= 0:
                stop_flag = True
                break

            new_coefs = Coefs(c_new)
            tol_check = np.linalg.norm(coefs.get_coefs() - c_new)

            if alpha == 1 and tol_check <= sig.n * config.GP_tol:
                stop_flag = True

            new_sig = atoms.calc_sig(new_coefs)
            f_new = f(new_sig)

            if f_new <= f_val + c1 * grad_diff:
                break

            alpha *= alpha_red

        if stop_flag or alpha < alpha_min:
            # print("2")
            return coefs

        coefs = new_coefs
        sig = new_sig
        f_val = f_new
        grad = f.calc_grad(sig)

    # print("3")
    return coefs


def Enhancemnet_step2(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    alpha = 0.01
    stop_flag = False
    iter = 0
    while not stop_flag and iter < config.GP_iters:
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)
        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        # Projection onto the simplex
        c = coefs.get_coefs()
        c_new = project_to_simplex(c - alpha * g, config.tau)
        # c_new = c - alpha * g
        coefs.update_coefficients(c_new)

        # Check convergence tolerance
        tol_check = np.linalg.norm(c - c_new)
        if tol_check <= config.GP_tol:
            stop_flag = True

        iter += 1

    # atoms.update_coefficients(coefs=c)
    return coefs


def Truncate(atoms: Atoms, f: GraphFunc, coefs: Coefs, F_th):
    """
    Truncates the given set of atoms and coefficients based on a threshold function value.

    Parameters:
    atoms (Atoms): The set of atoms to be truncated.
    f (GraphFunc): A function that evaluates the significance of the atoms.
    coefs (Coefs): The coefficients associated with the atoms.
    F_th (float): The threshold value for the function evaluation.

    Returns:
    tuple: A tuple containing the truncated set of atoms and the updated coefficients.
    """
    terminate = False
    while not terminate:
        remove_idx = remove_atom(atoms, f, coefs)

        new_atoms = deepcopy(atoms)
        new_coefs = deepcopy(coefs)
        new_atoms.remove_atom(new_atoms[remove_idx])
        new_coefs.remove_coef(remove_idx)

        if len(new_coefs) > 0:
            new_c = project_to_simplex(new_coefs, config.tau)
            new_coefs.update_coefficients(new_c)
            new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
            new_sig = new_atoms.calc_sig(new_coefs)
            f_new_sig = f(new_sig)

            if f_new_sig <= F_th:
                atoms = new_atoms
                coefs = new_coefs
            else:
                terminate = True
        else:
            terminate = True

    return atoms, coefs


def Matrix_Truncate(atoms: Atoms, f: GraphFunc, coefs: Coefs, F_th):
    sig = atoms.calc_sig(coefs)

    graph = Graph(sig.n, train_ratio=1)
    graph.A = sig.L
    graph.Abar = 2 * sig.L - 1
    graph.cov = f.graph.cov
    graph.y = np.argmax(sig.pi, axis=1)
    graph.train_mask = f.graph.train_mask
    graph.num_nodes = sig.n
    graph.num_features = sig.d
    graph.num_classes = sig.c

    new_f = GraphFunc(graph=graph)
    new_atoms = Atoms()
    new_coefs = Coefs()
    new_sig = EmptySig(sig.n, sig.d, sig.c)

    for itr in range(len(atoms)):
        new_atoms, new_coefs = Forward_step(new_atoms, new_f, new_coefs, new_sig, itr)
        new_coefs = Enhancemnet_step(new_atoms, new_f, new_coefs)

        new_sig = new_atoms.calc_sig(new_coefs)
        f_sig = f(new_sig)

        if f_sig < F_th:
            return new_atoms, new_coefs

    return atoms, coefs
