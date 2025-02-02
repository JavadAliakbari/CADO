from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


from src.Atom.coefs import Coefs
from src.Atom.atoms import Atoms
from src.Graph_Atom.graph_atom import GraphAtom
from src.Graph_Atom.graph_sig import GraphSig
from src.Graph_Atom.graph_func import GraphFunc
from src.utils.utils import *


def find_next_atom(grad: GraphSig, constraints, itr) -> GraphAtom:
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
        epsilon, λ = update_e(grad, atom, constraints, it, itr)
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


def find_next_atom2(grad: GraphSig, R, itr) -> GraphAtom:
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
    atom.R = R

    # Alternating minimization
    max_iter = config.iter_or_cycles_to_find_next_atom
    if config.mu1 == 0 and config.mu2 == 0:
        max_iter = 1

    th_old = np.inf
    plt.figure()
    th_list = []
    for it in range(max_iter):
        epsilon = update_e(grad, atom, it, itr)
        atom.update_e(epsilon)
        # th = grad.dot(atom.calc_sig_term())
        # diff = th_old - th
        # th_old = th

        # if config.mu1 != 0:
        #     R = update_R(grad, atom, it, itr)
        #     atom.update_R(R)
        # th = grad.dot(atom.calc_sig_term())
        # diff = th_old - th
        # th_old = th

        if config.mu2 != 0:
            b = update_b(grad, atom)
            atom.update_b(b)
            # th = grad.dot(atom.calc_sig_term())
            # diff = th_old - th
            # th_old = th

        th = grad.dot(atom.calc_sig_term())
        th_list.append(th)
        diff = th_old - th
        if abs(diff) < 1e-6:
            break
        else:
            th_old = th

    # Return the next atom in the form [epsilon @ epsilon^T, R, b]
    if plot:
        sub_folder = f"{save_path}{itr}/"
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        plt.figure()
        plt.plot(th_list[1:])
        plt.savefig(f"{sub_folder}th_list.jpg", format="jpg")
    return atom


def find_eps_lambda(A, constraints):
    n = A.shape[0]
    λ, evecs_large = sc.linalg.eigh(A, subset_by_index=(n - 1, n - 1))
    x = evecs_large[:, 0]

    epsilon_pri = 1e-10
    max_iterations = 1000

    for _ in range(max_iterations):
        h = λ * (x**2)
        # Update y
        y = np.clip(h, 0, constraints)

        e = np.sign(x) * np.sqrt(y / (λ + 1e-10))
        e /= np.linalg.norm(e) + 1e-10
        λ = np.einsum("i,ij,j->", e, A, e)

        r = y - h

        # Check convergence
        if np.linalg.norm(r) <= epsilon_pri:
            break

    return e, λ


def find_eps_lambda2(A, constraints):
    # Initialize variables
    # x = initial_x  # Ensure x^T x = 1
    n = A.shape[0]
    nu = np.zeros(n)
    rho = 5  # Penalty parameter

    epsilon_pri = 1e-6
    max_iterations = 1000

    for k in range(max_iterations):
        # Update x
        M = A + np.diag(nu)
        λ, evecs_large = sc.linalg.eigh(M, subset_by_index=(n - 1, n - 1))
        x = evecs_large[:, 0]
        # eigenvalues, eigenvectors = np.linalg.eigh(M)
        # λ = np.argmax(eigenvalues)
        # x = eigenvectors[:, λ]

        h = λ * (x**2)
        # Update y
        y = np.clip(h - nu / rho, 0, constraints)

        # Update dual variables
        r = y - h
        nu += rho * r

        # Check convergence
        if np.linalg.norm(r) <= epsilon_pri:
            break

    # e = np.sign(x) * np.sqrt(y / λ)
    # e =
    # λ2 = np.einsum("i,ij,j->", e, A, e)
    return x, λ


## Step 1: Fix R and b, update epsilon
def update_e(grad: GraphSig, atom: GraphAtom, constraints, iter=0, itr=0):
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

    epsilon, λ = find_eps_lambda(-tmp3, constraints)

    # A = -tmp3
    # λ, evecs_large = sc.linalg.eigh(A, subset_by_index=(N - 1, N - 1))
    # epsilon = evecs_large[:, 0]

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
        return 0, np.inf
    sig = atoms.calc_sig(coefs)

    min_val = np.inf
    arg_min_idx = -1
    for idx in range(len(coefs)):
        # new_coefs = deepcopy(coefs)
        # new_atoms = deepcopy(atoms)
        # new_coefs.remove_coef(idx)
        # # new_atoms.remove_atom(atoms[idx])
        # new_atoms.remove_idx(idx)
        # c_new = project_to_simplex(new_coefs, config.tau)
        # new_coefs = Coefs(c_new)
        # if config.enhance:
        #     new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
        # residual_sig = new_atoms.calc_sig(new_coefs)
        single_sig = atoms[idx].calc_sig_term()
        if coefs[idx] == 0:
            residual_sig = sig
        else:
            single_sig = coefs[idx] * single_sig
            residual_sig = sig - single_sig
        val = f(residual_sig)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx, min_val


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
    constraints = 1 - sum(
        [c * (atom.e**2) for c, atom in zip(coefs.get_coefs(), atoms)],
        np.zeros(300),
    )
    constraints[constraints < 0] = 0

    grad = f.calc_grad(sig)

    new_atom, λ = find_next_atom(grad, constraints, itr)
    coefs.add_coef(λ)
    atoms.add_atom(new_atom)

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
    # if len(coefs) <= 1:
    #     coefs.update_coefficients([config.tau])
    #     return coefs

    alpha_min, alpha_red, c1 = 1e-6, 0.5, 1e-3
    E = [atom.e for atom in atoms]
    E2 = np.array(E).T ** 2
    c_new = project_inside_simplex2(coefs.get_coefs(), E2, 1)
    coefs = Coefs(c_new)

    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)

    for _ in range(config.GP_iters):
        grad_λ = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms])
        alpha, stop_flag = 1, False

        while alpha > alpha_min and not stop_flag:
            new_λ = coefs.get_coefs() - alpha * grad_λ
            c_new = project_inside_simplex2(new_λ, E2, 1)
            new_coefs = Coefs(c_new)
            tol_check = np.linalg.norm(coefs.get_coefs() - c_new)

            if alpha == 1 and tol_check <= config.GP_tol:
                stop_flag = True

            if f(atoms.calc_sig(new_coefs)) <= f(sig) + c1 * (
                (c_new - coefs.get_coefs()).T @ grad_λ
            ):
                break

            alpha *= alpha_red

        if stop_flag or alpha < alpha_min:
            return coefs

        coefs = deepcopy(new_coefs)
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)

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
        remove_idx, f_val = remove_atom(atoms, f, coefs)
        if f_val <= F_th:
            atoms.remove_atom(atoms[remove_idx])
            coefs.remove_coef(remove_idx)
        else:
            terminate = True

    return atoms, coefs


def Enhance_step(atoms, coefs):
    E = [atom.e for atom in atoms]
    E = np.array(E).T
    n, c = E.shape

    # l = n * np.random.rand(c)
    # l = n * np.ones(c) / c
    l = coefs.get_coefs()
    l = project_to_simplex2(l, n)

    total = 1000
    # bar = tqdm(total=total)
    for _ in range(total):
        H = E**2 @ np.diag(l)
        A = project_to_simplex2(H, 1)
        new_l = np.sum(A, axis=0)

        # f1 = sum(sum(np.abs(H - A)))
        f2 = sum(np.abs(new_l - l))
        l = new_l

        if f2 < 1e-6:
            break

        # bar.set_postfix(f1=f1, f2=f2)
        # bar.update()

    new_E = A @ np.diag(1 / l)
    for idx, atom in enumerate(atoms):
        atom.e = np.sqrt(new_E[:, idx])

    coefs.update_coefficients(l)

    return atoms, coefs
