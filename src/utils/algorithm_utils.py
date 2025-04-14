import numpy as np
from sklearn.cluster import SpectralClustering

from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_atom import GraphAtom
from utils.graph import Graph
from utils.utils import *


def Piv2EC(Piv):
    c = np.sum(Piv, axis=0)
    E2 = Piv / (c[None, :] + 1e-10)
    E = np.sqrt(E2)

    return E, c


def map_to_simplex(atoms: Atoms, coefs: Coefs):
    Piv = np.array([coef * atom.e**2 for coef, atom in zip(coefs, atoms)]).T
    Piv = project_to_simplex(Piv, 1)
    E, c = Piv2EC(Piv)
    coefs.update_coefficients(c)
    for idx, atom in enumerate(atoms):
        atom.e = E[:, idx]

    return atoms, coefs


def interpolate(atoms1, coefs1, atoms2, coefs2, alpha):
    alpha = float(alpha)
    coefs = (1 - alpha) * coefs1 + alpha * coefs2
    alpha_i = alpha * coefs2 / coefs
    atoms = (1 - alpha_i) * atoms1 + alpha_i * atoms2
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


def create_similar_atoms(n, d, c, num_atoms):
    atoms = Atoms()
    coefs = Coefs()
    for _ in range(num_atoms):
        atom = GraphAtom()
        atom.e = np.ones(n)
        atom.R = np.eye(d)
        atom.b = np.ones(c)
        atom.project_to_bound()
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
        RR = project_to_psd_cone(RR, data.rho_min, data.rho_max)

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
        alpha_high = 0.2  # Much wider than the original 0.1

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
