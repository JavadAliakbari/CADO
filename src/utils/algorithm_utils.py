import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import root

from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_atom import GraphAtom
from utils.graph import Graph
from utils.utils import *


def map_to_simplex(atoms: Atoms):
    Piv = np.array([atom.e**2 for atom in atoms]).T
    Piv = project_to_simplex(Piv, 1)
    E = np.sqrt(Piv)
    for idx, atom in enumerate(atoms):
        atom.e = E[:, idx]

    return atoms


def interpolate(atoms1, atoms2, alpha):
    alpha = float(alpha)
    # alpha_c = alpha / len(coefs1)
    # coefs = (1 - alpha) * coefs1 + alpha * coefs2
    # alpha_i = alpha * coefs2 / coefs
    # atoms = (1 - alpha_i) * atoms1 + alpha_i * atoms2

    atoms = Atoms()
    # alpha2 = 1 * alpha
    for atom1, atom2 in zip(atoms1, atoms2):
        atom = GraphAtom()
        atom.e = np.sqrt((1 - alpha) * (atom1.e**2) + alpha * (atom2.e**2))

        # Linear interpolation of theta (k-means centers)
        atom.theta = (1 - alpha) * atom1.theta + alpha * atom2.theta
        # Project theta to satisfy norm constraint
        atom.theta = project_to_ball(atom.theta, data.tau)
        # atom.theta = np.sqrt((1 - a_i) * (atom1.theta**2) + a_i * (atom2.theta**2))

        atom.b = (1 - alpha) * atom1.b + alpha * atom2.b

        atoms.add_atom(atom)

    # for a_i, atom, atom1, atom2 in zip(alpha_i, atoms, atoms1, atoms2):
    #     atom.R = geodesic_interpolation(atom1.R, atom2.R, a_i)
    #     atom.R = project_to_psd_cone(atom.R, data.rho_min, data.rho_max)
    # # # atoms, coefs = map_to_simplex(atoms, coefs)

    return atoms


def create_random_atoms(n, d, c, num_atoms):
    atoms = Atoms()
    for _ in range(num_atoms):
        atom = GraphAtom()
        atom.initialize(n, d, c)
        atoms.add_atom(atom)

    return atoms


def create_similar_atoms(graph: Graph, num_atoms):
    atoms = Atoms()
    n = graph.num_nodes
    d = graph.num_features
    for _ in range(num_atoms):
        atom = GraphAtom()
        atom.e = np.sqrt(n) * np.ones(n) / num_atoms
        # Initialize theta as a random vector with norm <= tau
        # atom.theta = np.random.randn(d)
        # atom.theta = project_to_ball(atom.theta, data.tau)
        atom.theta = np.zeros(d)
        atom.b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        atom.b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", atom.e**2, atom.b)
        atom.b = np.zeros(graph.num_classes, dtype=float)
        atom.b[np.argmax(a)] = 1

        atoms.add_atom(atom)

    return atoms


def create_optimal_atoms(graph: Graph):
    atoms = Atoms()
    E = graph.E
    c = graph.num_classes
    n = graph.num_nodes

    for i in range(c):
        atom0 = GraphAtom()
        e = E[:, i]

        # Compute optimal cluster center as weighted mean of features
        weights = e**2  # (n,)
        theta = np.einsum("i,ij->j", weights, graph.X)  # Weighted mean: (d,)
        theta = theta / np.sum(weights)  # Normalize by sum of weights

        # Project theta to satisfy norm constraint
        theta = project_to_ball(theta, data.tau)

        b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", e**2, b)
        b = np.zeros(graph.num_classes, dtype=float)
        b[np.argmax(a)] = 1

        atom0.set_params(e, theta, b)
        atoms.add_atom(atom0)
    return atoms


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
    d = graph.num_features
    for i in range(graph.num_clusters):
        atom = GraphAtom()
        # atom.initialize(graph.num_nodes, graph.num_features, graph.num_clusters)
        mask = cluster_labels == i
        coef = np.sum(mask)
        atom.e = np.zeros(n)
        atom.e[mask] = 1
        atom.e /= np.linalg.norm(atom.e)

        # Compute cluster center as weighted mean of features
        cluster_features = graph.X[mask]
        if len(cluster_features) > 0:
            atom.theta = np.mean(cluster_features, axis=0)
        else:
            atom.theta = np.random.randn(d)

        # Project theta to satisfy norm constraint
        atom.theta = project_to_ball(atom.theta, data.tau)

        b = np.zeros((n, graph.num_clusters))
        node_ids = np.arange(n)
        b[node_ids[graph.train_mask], graph.y[graph.train_mask]] = 1
        a = np.einsum("i, ij-> j", atom.e**2, b)
        atom.b = np.zeros(graph.num_classes, dtype=float)
        atom.b[np.argmax(a)] = 1
        atoms.add_atom(atom)
        coefs.add_coef(coef)
    return atoms, coefs


def line_search(atoms1, atoms2, f, alpha_high_limit=1, max_steps=10, tol=1e-5):
    """
    Perform line search to find the optimal step size alpha that minimizes f.

    Args:
        atoms1: Current atoms.
        atoms2: New atoms (direction).
        f: Objective function to minimize.
        max_steps: Maximum number of steps for the line search.
        tol: Tolerance for convergence.

    Returns:
        atoms: Updated atoms after applying the optimal step size.
        coefs: Updated coefficients after applying the optimal step size.
        alpha: Optimal step size.
    """

    alpha_low = 0
    alpha_high = alpha_high_limit

    atoms_low = interpolate(atoms1, atoms2, alpha_low)
    sig_low = atoms_low.calc_sig()
    f1 = f(sig_low)

    atoms_high = interpolate(atoms1, atoms2, alpha_high)
    sig_high = atoms_high.calc_sig()
    f2 = f(sig_high)

    # Perform the golden-section search
    for _ in range(max_steps):
        if abs(alpha_high - alpha_low) < tol:
            break

        if f1 > f2:
            alpha_low = 0.5 * (alpha_high + alpha_low)
            atoms_low = interpolate(atoms1, atoms2, alpha_low)
            sig_low = atoms_low.calc_sig()
            f1 = f(sig_low)
        else:
            alpha_high = 0.5 * (alpha_high + alpha_low)

            atoms_high = interpolate(atoms1, atoms2, alpha_high)
            sig_high = atoms_high.calc_sig()
            f2 = f(sig_high)

    # Choose the best alpha
    if f1 < f2:
        alpha = alpha_low
    else:
        alpha = alpha_high

    # Update atoms and coefficients using the optimal alpha
    atoms = interpolate(atoms1, atoms2, alpha)

    return atoms


def schedule_lr(
    ii,
    max_iter=algorithm.iters,
    min_lr=optimizer.min,
    max_lr=optimizer.lr,
    warmup_steps=algorithm.warmup_steps,
    decay_steps=optimizer.decay_steps,
    decay_rate=optimizer.decay_rate,
    method=optimizer.method,
    # method="exponential",
    # method="cosine",
):
    # return 2 / (ii + 2)

    if method == "cosine":
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(np.pi * (ii - warmup_steps) / (max_iter - warmup_steps))
        )
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
    elif method == "linear":
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
        else:
            lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
    elif method == "exponential":
        lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
        elif ii < max_iter - decay_steps:
            lr = max_lr
        else:
            lr = max_lr + (min_lr - max_lr) * np.exp(
                -decay_rate * (ii - max_iter + decay_steps)
            )
    return lr
