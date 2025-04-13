from copy import deepcopy
import numpy as np
from sklearn.cluster import SpectralClustering

from Atom.coefs import Coefs
from Atom.atoms import Atoms
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
    λ = atom.conditional_gradient_update(grad, itr)

    return atom, λ


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
    gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma, id=len(coefs) - 1)

    return atoms, coefs


def update_coefs(atoms: Atoms, f: GraphFunc, coefs: Coefs, sig, lr, itr):
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

    for _ in range(itr):
        grad = f.calc_grad(sig)
        g = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms])
        c_new = project_to_simplex(coefs.get_coefs() - lr * g, config.tau)
        coefs.update_coefficients(c_new)
        sig = atoms.calc_sig(coefs)

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

    alpha_min, alpha_red, c1 = 0.01, 0.5, 1e-3
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

            if alpha == sig.n and tol_check <= sig.n * config.GP_tol:
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


def Enhancemnet_step2(
    atoms: Atoms,
    f: GraphFunc,
    coefs: Coefs,
    iters=config.GP_iters,
    lr=config.optimizer.lr,
):
    stop_flag = False
    iter = 0
    while not stop_flag and iter < iters:
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)
        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        # Projection onto the simplex
        c = coefs.get_coefs()
        c_new = project_to_simplex(c - lr * g, config.tau)
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

    return atoms, coefs, f


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


def create_sig(E, R, b, λ):
    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    for i in range(b.shape[1]):

        atom0 = GraphAtom()
        S, U = np.linalg.eigh(R[i])
        mask = S >= config.sig_rho_max - 0.0000001
        S[mask] = config.rho_max
        S[~mask] = config.rho_min

        RR = U @ np.diag(S) @ U.T
        atom0.set_params(E[:, i], RR, b[i])
        atoms.add_atom(atom0)
        coefs.add_coef(λ[i])
    sig = atoms.calc_sig(coefs)
    return sig


def create_random_atoms(n, d, c, num_atoms):
    atoms = Atoms()
    coefs = Coefs()
    for _ in range(num_atoms):
        atom = GraphAtom()
        atom.initialize(n, d, c)
        atoms.add_atom(atom)

    coefs = n * np.random.rand(num_atoms)
    coefs = project_to_simplex(coefs, n)
    coefs = Coefs(coefs)
    return atoms, coefs


def create_optimal_atoms(graph: Graph):
    atoms = Atoms()
    coefs = Coefs()
    E = graph.E
    # E[np.abs(E) < 0.00001] = 0.00001
    # R = graph.Ri_matrices
    # b = graph.b
    c = graph.num_classes
    Rv = graph.cov
    n = graph.num_nodes

    for i in range(c):
        atom0 = GraphAtom()
        # RR = project_to_psd_cone(R[i], config.rho_min, config.rho_max)
        e = E[:, i]
        RR = np.einsum("ijk,i->jk", Rv, e**2)
        RR = project_to_psd_cone(RR, config.rho_min, config.rho_max)

        b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", e**2, b)
        b = np.zeros(graph.num_classes, dtype=float)
        b[np.argmax(a)] = 1

        atom0.set_params(e, RR, b)
        atoms.add_atom(atom0)
        coefs.add_coef(graph.λ[i])
    return atoms, coefs


def run_spectral_clustering(graph, n_clusters):
    """
    Run spectral clustering on the graph's adjacency matrix.

    Args:
        graph (SBM): The graph object.
        n_clusters (int): The number of clusters.

    Returns:
        np.ndarray: The cluster labels for each node.
    """
    adjacency_matrix = graph.A
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
    )
    labels = spectral.fit_predict(adjacency_matrix)
    return labels


def create_atoms_from__cluster_labels(graph: Graph, cluster_labels):
    atoms = Atoms()
    coefs = Coefs()
    n = graph.num_nodes
    Rv = graph.cov
    for i in range(graph.num_clusters):
        atom = GraphAtom()
        # atom.initialize(graph.num_nodes, graph.num_features, graph.num_clusters)
        mask = cluster_labels == i
        coef = np.sum(mask)
        atom.e = np.zeros(n)
        atom.e[mask] = 1
        atom.e /= np.linalg.norm(atom.e)
        atom.R = np.einsum("ijk,i->jk", Rv, atom.e**2)
        atom.R = project_to_psd_cone(atom.R, config.rho_min, config.rho_max)

        b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", atom.e**2, b)
        atom.b = np.zeros(graph.num_classes, dtype=float)
        atom.b[np.argmax(a)] = 1
        atoms.add_atom(atom)
        coefs.add_coef(coef)
    return atoms, coefs
