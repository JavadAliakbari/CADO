import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import root

from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_atom import GraphAtom
from utils.graph import Graph
from utils.utils import *


def Piv2EC(Piv, s=None):
    c = np.sum(Piv, axis=0)
    E2 = Piv / (c[None, :] + 1e-10)
    E = np.sqrt(E2)
    # if s is not None:
    #     E *= s

    return E, c


def map_to_simplex(atoms: Atoms, coefs: Coefs):
    s = np.array([np.sign(atom.e) for atom in atoms]).T
    Piv = np.array([coef * atom.e**2 for coef, atom in zip(coefs, atoms)]).T
    Piv = project_to_simplex(Piv, 1)
    E, c = Piv2EC(Piv, s)
    coefs.update_coefficients(c)
    for idx, atom in enumerate(atoms):
        atom.e = E[:, idx]

    return atoms, coefs


def interpolate(atoms1, coefs1, atoms2, coefs2, alpha):
    alpha = float(alpha)
    # alpha_c = alpha / len(coefs1)
    coefs = (1 - alpha) * coefs1 + alpha * coefs2
    alpha_i = alpha * coefs2 / coefs
    # atoms = (1 - alpha_i) * atoms1 + alpha_i * atoms2

    atoms = Atoms()
    alpha2 = 1 * alpha
    for a_i, atom1, atom2 in zip(alpha_i, atoms1, atoms2):
        atom = GraphAtom()
        atom.e = np.sqrt((1 - a_i) * (atom1.e**2) + a_i * (atom2.e**2))
        # atom.e = np.sign(atom1.e) * np.sqrt(
        #     (1 - a_i) * (atom1.e**2) + a_i * (atom2.e**2)
        # )
        atom.R = (1 - alpha2) * atom1.R + alpha2 * atom2.R
        # atom.R = geodesic_interpolation(atom1.R, atom2.R, alpha2)
        # atom.R = harmonic_interp(atom1.R, atom2.R, alpha2)
        # atom.R = log_euclidean_interp(atom1.R, atom2.R, alpha2)
        # atom.R = project_to_psd_cone(atom.R, data.rho_min, data.rho_max)
        # atom.R = project_to_unit_trace(atom.R, data.rho_min, data.rho_max)
        atom.b = (1 - alpha2) * atom1.b + alpha2 * atom2.b

        atoms.add_atom(atom)

    # for a_i, atom, atom1, atom2 in zip(alpha_i, atoms, atoms1, atoms2):
    #     atom.R = geodesic_interpolation(atom1.R, atom2.R, a_i)
    #     atom.R = project_to_psd_cone(atom.R, data.rho_min, data.rho_max)
    # # # atoms, coefs = map_to_simplex(atoms, coefs)

    return atoms, coefs


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


def create_similar_atoms(graph: Graph, num_atoms):
    atoms = Atoms()
    coefs = Coefs()
    c = graph.num_classes
    Rv = graph.cov
    n = graph.num_nodes
    d = graph.num_features
    for _ in range(num_atoms):
        atom = GraphAtom()
        atom.e = np.ones(n)
        RR = np.einsum("ijk,i->jk", Rv, atom.e**2)
        RR = project_to_psd_cone(RR, data.rho_min, data.rho_max)
        atom.R = np.eye(d)
        atom.b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        atom.b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", atom.e**2, atom.b)
        atom.b = np.zeros(graph.num_classes, dtype=float)
        atom.b[np.argmax(a)] = 1

        atoms.add_atom(atom)

    coefs = np.ones(num_atoms)
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
        # RR = project_to_psd_cone(R[i], data.rho_min, data.rho_max)
        e = E[:, i]
        RR = np.einsum("ijk,i->jk", Rv, e**2)
        RR = project_to_unit_trace(RR, data.rho_min, data.rho_max)

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
        atom.R = project_to_psd_cone(atom.R, data.rho_min, data.rho_max)

        b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", atom.e**2, b)
        atom.b = np.zeros(graph.num_classes, dtype=float)
        atom.b[np.argmax(a)] = 1
        atoms.add_atom(atom)
        coefs.add_coef(coef)
    return atoms, coefs


def line_search(atoms1, coefs1, atoms2, coefs2, f, max_steps=10, tol=1e-5):
    """
    Perform line search to find the optimal step size alpha that minimizes f.

    Args:
        atoms1: Current atoms.
        coefs1: Current coefficients.
        atoms2: New atoms (direction).
        coefs2: New coefficients (direction).
        f: Objective function to minimize.
        max_steps: Maximum number of steps for the line search.
        tol: Tolerance for convergence.

    Returns:
        atoms: Updated atoms after applying the optimal step size.
        coefs: Updated coefficients after applying the optimal step size.
        alpha: Optimal step size.
    """

    alpha_low = 0
    alpha_high = 0.05

    atoms_low, coefs_low = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_low)
    sig_low = atoms_low.calc_sig(coefs_low)
    f1 = f(sig_low)

    atoms_high, coefs_high = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_high)
    sig_high = atoms_high.calc_sig(coefs_high)
    f2 = f(sig_high)

    # Perform the golden-section search
    for _ in range(max_steps):
        if abs(alpha_high - alpha_low) < tol:
            break

        if f1 > f2:
            alpha_low = 0.5 * (alpha_high + alpha_low)
            atoms_low, coefs_low = interpolate(
                atoms1, coefs1, atoms2, coefs2, alpha_low
            )
            sig_low = atoms_low.calc_sig(coefs_low)
            f1 = f(sig_low)
        else:
            alpha_high = 0.5 * (alpha_high + alpha_low)

            atoms_high, coefs_high = interpolate(
                atoms1, coefs1, atoms2, coefs2, alpha_high
            )
            sig_high = atoms_high.calc_sig(coefs_high)
            f2 = f(sig_high)

    # Choose the best alpha
    if f1 < f2:
        alpha = alpha_low
    else:
        alpha = alpha_high

    # Update atoms and coefficients using the optimal alpha
    atoms, coefs = interpolate(atoms1, coefs1, atoms2, coefs2, alpha)

    return atoms, coefs


def line_search2(
    atoms1, coefs1, atoms2, coefs2, f, max_steps=20, tol=1e-5, num_restarts=3
):
    """
    Perform enhanced line search to find the optimal step size alpha that minimizes f.
    Uses multiple techniques to avoid getting stuck in local minima:
    - Wider initial search range
    - Adaptive range expansion
    - Multiple random restarts
    - Golden section search

    Args:
        atoms1: Current atoms.
        coefs1: Current coefficients.
        atoms2: New atoms (direction).
        coefs2: New coefficients (direction).
        f: Objective function to minimize.
        max_steps: Maximum number of steps for each search.
        tol: Tolerance for convergence.
        num_restarts: Number of random restarts to try.

    Returns:
        atoms: Updated atoms after applying the optimal step size.
        coefs: Updated coefficients after applying the optimal step size.
    """
    best_alpha = 0
    best_value = float("inf")

    # Try multiple random initializations to avoid local minima
    for _ in range(num_restarts):
        # Start with a wider search range
        alpha_low = 0
        alpha_high = 0.5  # Much wider than the original 0.1

        # Add some randomization to the high value to increase exploration
        alpha_high = min(1.0, alpha_high * (0.5 + np.random.rand()))

        atoms_low, coefs_low = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_low)
        sig_low = atoms_low.calc_sig(coefs_low)
        f_low = f(sig_low)

        atoms_high, coefs_high = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_high)
        sig_high = atoms_high.calc_sig(coefs_high)
        f_high = f(sig_high)

        # Golden section search (more efficient than bisection)
        golden_ratio = (np.sqrt(5) - 1) / 2  # ≈ 0.618

        # Initialize intermediate points
        alpha_mid1 = alpha_low + (1 - golden_ratio) * (alpha_high - alpha_low)
        alpha_mid2 = alpha_low + golden_ratio * (alpha_high - alpha_low)

        atoms_mid1, coefs_mid1 = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_mid1)
        sig_mid1 = atoms_mid1.calc_sig(coefs_mid1)
        f_mid1 = f(sig_mid1)

        atoms_mid2, coefs_mid2 = interpolate(atoms1, coefs1, atoms2, coefs2, alpha_mid2)
        sig_mid2 = atoms_mid2.calc_sig(coefs_mid2)
        f_mid2 = f(sig_mid2)

        # Keep track of all evaluated points in case we need to check boundaries
        evaluated_points = [
            (alpha_low, f_low),
            (alpha_high, f_high),
            (alpha_mid1, f_mid1),
            (alpha_mid2, f_mid2),
        ]

        # Perform search
        for _ in range(max_steps):
            if abs(alpha_high - alpha_low) < tol:
                break

            if f_mid1 > f_mid2:
                # Move the left part of the interval
                alpha_low = alpha_mid1
                f_low = f_mid1
                alpha_mid1 = alpha_mid2
                f_mid1 = f_mid2

                # Calculate new right mid point
                alpha_mid2 = alpha_low + golden_ratio * (alpha_high - alpha_low)
                atoms_mid2, coefs_mid2 = interpolate(
                    atoms1, coefs1, atoms2, coefs2, alpha_mid2
                )
                sig_mid2 = atoms_mid2.calc_sig(coefs_mid2)
                f_mid2 = f(sig_mid2)
                evaluated_points.append((alpha_mid2, f_mid2))
            else:
                # Move the right part of the interval
                alpha_high = alpha_mid2
                f_high = f_mid2
                alpha_mid2 = alpha_mid1
                f_mid2 = f_mid1

                # Calculate new left mid point
                alpha_mid1 = alpha_low + (1 - golden_ratio) * (alpha_high - alpha_low)
                atoms_mid1, coefs_mid1 = interpolate(
                    atoms1, coefs1, atoms2, coefs2, alpha_mid1
                )
                sig_mid1 = atoms_mid1.calc_sig(coefs_mid1)
                f_mid1 = f(sig_mid1)
                evaluated_points.append((alpha_mid1, f_mid1))

        # Check if we need to explore beyond our initial range
        # If the best point is at the boundary, expand in that direction
        evaluated_points.sort(key=lambda x: x[1])  # Sort by function value
        best_alpha_local, best_value_local = evaluated_points[0]

        # If this is better than our previous best, update
        if best_value_local < best_value:
            best_value = best_value_local
            best_alpha = best_alpha_local

    # Final interpolation with the best alpha found
    atoms, coefs = interpolate(atoms1, coefs1, atoms2, coefs2, best_alpha)
    return atoms, coefs


def find_diagonal_scaling(eps, c_new, max_iter=1000, tol=1e-12):
    """
    Solve for the diagonal entries t_v > 0 (v=1..n) that satisfy:
        sum_{i=1..k} c_new[i] * (t_v^2 * eps[i,v]^2) / (sum_{u=1}^n t_u^2 * eps[i,u]^2) = 1  for each v=1..n.

    Parameters:
    -----------
    eps     : (k, n) array
              The original k unit vectors eps_i in R^n (row-wise).
    c_new   : (k,) array
              The new coefficients that sum to n.
    max_iter: int
              Maximum number of fixed-point iterations to perform.
    tol     : float
              Convergence tolerance on the change in t_v from one iteration to the next.

    Returns:
    --------
    t       : (n,) array
              The converged solution for the diagonal scaling factors.
    """
    n, k = eps.shape

    # Initialize t_v = 1.0 as a simple starting guess
    t = np.ones(n, dtype=np.float64)

    # bar = tqdm(total=max_iter)
    for _ in range(max_iter):
        # Compute G_i(t) = sum_{u=1}^n t_u^2 * eps[i,u]^2 for each i
        # shape: (k,)
        G = np.sum(t[:, None] * eps, axis=0)

        # Update each t_v via fixed-point formula
        t_new = 1.0 / np.sum(c_new[None, :] * eps / G[None, :], axis=1)

        # Check convergence
        loss = np.max(np.abs(t_new - t))
        # bar.set_postfix(loss=loss)
        # bar.update(1)
        if loss < tol:
            t = t_new
            return t

        raise NotImplementedError

    return t


def construct_tilde_epsilon(eps, t):
    """
    Given eps (k x n) and the diagonal scaling vector t (n,),
    construct tilde_eps[i] = D * eps[i] / ||D * eps[i]||, where D = diag(t).

    Returns tilde_eps (k x n).
    """
    # (k,n) times (n,) [broadcast over columns], i.e. scale each column of eps by t
    scaled = eps * t[:, None]  # shape (k,n)

    # Norm of each row
    norms = np.sum(scaled, axis=0, keepdims=True)  # shape (k,1)

    return scaled / norms  # shape (k,n)


def F_system(x, eps, c_new):
    """
    Returns the vector F(t) of dimension n, where
       F_v(t) = sum_{i=1}^k [ c_new[i] * t_v^2 * eps[i,v]^2 / G_i(t ) ] - 1
    and G_i(t) = sum_{u=1}^n t_u^2 * eps[i,u]^2.

    Parameters
    ----------
    t      : (n,) array, the unknowns t_v (assumed > 0 ideally).
    eps    : (k, n) array, original vectors eps_i in R^n (row-wise).
    c_new  : (k,) array, new coefficients that sum to n.

    Returns
    -------
    F_val  : (n,) array, the residuals F_v(t).
    """
    n, k = eps.shape
    t = np.exp(x)
    # G_i(t) for each i
    # shape: (k,) => each G_i is sum_{u=1..n} t_u^2 * eps[i,u]^2
    G = np.sum(t[:, None] * eps, axis=0)

    F_val = np.zeros(n, dtype=np.float64)
    F_val = np.sum(c_new[None, :] * eps / G[None, :], axis=1) - 1
    return F_val


def solve_t_v(eps, c_new, x_init=None, method="hybr", **root_kwargs):
    """
    Solve for x_v in R, with t_v = exp(x_v) > 0, ensuring positivity.

    Uses scipy.optimize.root to solve F_system_in_log_space(x)=0.

    Parameters
    ----------
    eps        : (k, n) array
                 Original vectors eps_i in R^n (row i is eps_i).
    c_new      : (k,) array
                 New coefficients that sum to n.
    x_init     : (n,) array or None
                 Initial guess in log-space. If None, uses  log(1.0).
    method     : str
                 The method passed to `scipy.optimize.root` (e.g. 'hybr', 'lm', 'broyden1', 'krylov').
    **root_kwargs:
                 Additional keyword arguments for `scipy.optimize.root`.

    Returns
    -------
    t_sol      : (n,) array
                 The positive solution for t_v = exp(x_v).
    success    : bool
                 True if the solver succeeded.
    res        : OptimizeResult
                 Full result object from `root`.
    """
    k, n = eps.shape

    # If no initial guess, start with all zero in log-space => t_v=exp(0)=1
    if x_init is None:
        x_init = np.zeros(n, dtype=np.float64)

    # Solve for x via a root-finding method
    res = root(F_system, x_init, args=(eps, c_new), method=method, **root_kwargs)
    success = res.success

    # Convert x -> t
    t_sol = np.exp(res.x)

    return t_sol, success, res


def find_eps(E, c_new):
    n, k = E.shape
    eps = E**2
    # -- Solve for t:
    # t = find_diagonal_scaling(eps, c_new, max_iter=2000, tol=1e-14)
    x_init = np.log(np.ones(n))  # i.e. t=1 initially
    t, success, res = solve_t_v(eps, c_new, x_init=x_init, method="broyden1")
    if not success:
        raise RuntimeError(res.message)

    # -- Construct tilde_eps:
    tilde_eps = construct_tilde_epsilon(eps, t)

    Piv = c_new[None, :] * tilde_eps

    Piv = project_to_simplex(Piv, 1)
    tilde_eps, tilde_c = Piv2EC(Piv)

    return tilde_eps, tilde_c
