"""
Synthetic instances of Sec. 6: features with an exact cluster diameter rho and a
graph with an exact misconnection rate p_max (eq. (6)).

make_clusters_fixed_rho, _add_circulant, _add_bipartite and empirical_pmax are
the generators of the original experiment code (src/cogent_empirical_validation.py),
copied unchanged.  construct_controlled_graph is the modified graph generator
used for the final figures (see CHANGES.md, "Experiments").
"""

import numpy as np


# ------------------------------------------------------------------
# Features
# ------------------------------------------------------------------
def make_clusters_fixed_rho(rho, centroids, n_per_cluster, rng):
    """
    Place n_per_cluster points around each centroid so that the cluster
    diameter (max pairwise distance, Remark 4.5) equals `rho` exactly.

    Construction:  put n-2 random points uniformly on the sphere of radius
    rho/2 around the centroid, plus one antipodal pair on that sphere
    (so the diameter is rho and is attained).
    """
    K_local, d = centroids.shape
    X_list = []
    labels = []
    r = 0.5 * rho
    for k in range(K_local):
        # one antipodal pair guarantees the diameter is exactly rho
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u) + 1e-12
        pts = [centroids[k] + r * u, centroids[k] - r * u]
        # remaining points: uniform on sphere of radius r
        m = n_per_cluster - 2
        if m > 0:
            G = rng.standard_normal((m, d))
            G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
            pts.append(centroids[k] + r * G)
        Xi = np.vstack(
            [pts[0][None, :], pts[1][None, :]] + ([pts[2]] if len(pts) == 3 else [])
        )
        X_list.append(Xi)
        labels.extend([k] * n_per_cluster)
    X = np.vstack(X_list)
    labels = np.array(labels, dtype=int)
    return X, labels


# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------
def _add_circulant(A, nodes, deg):
    """Add a circulant of degree `deg` among `nodes` (in-place).

    Uses offsets +/-1, +/-2, ..., for the floor(deg/2) closest pairs.
    If `deg` is odd, requires len(nodes) to be even and additionally adds
    the antipodal offset N/2 (a perfect matching).  Net degree = `deg`
    on every node.
    """
    n = len(nodes)
    if deg <= 0 or n < 2:
        return
    half = deg // 2
    for off in range(1, half + 1):
        for k in range(n):
            u, v = nodes[k], nodes[(k + off) % n]
            A[u, v] = A[v, u] = 1.0
    if deg % 2 == 1:
        # antipodal matching -- only valid for even n
        if n % 2 != 0:
            raise ValueError(
                f"Cannot build a {deg}-regular circulant on {n} nodes " f"(odd n*deg)."
            )
        m = n // 2
        for k in range(m):
            u, v = nodes[k], nodes[k + m]
            A[u, v] = A[v, u] = 1.0


def _add_bipartite(A, rows, cols, row_deg):
    """Add a regular bipartite graph: each row-node has `row_deg` edges to cols
    and (when |rows|==|cols|) each col-node also has `row_deg`."""
    Nr, Nc = len(rows), len(cols)
    if row_deg <= 0 or Nr == 0 or Nc == 0:
        return
    for s in range(row_deg):
        for k in range(Nr):
            u, v = rows[k], cols[(k + s) % Nc]
            A[u, v] = A[v, u] = 1.0


def construct_controlled_graph(labels, p_target, rng=None):
    """Regular graph with p_max = round(p_target * N) / N exactly (delta_ij = 0).

    Every class is a circulant (intra-class degree intra_deg) and every pair of
    classes is joined by a shifted regular bipartite graph (m_inter edges per
    node towards every other class).  Then p_ij = m_inter/N for i != j and
    p_ii = (N - intra_deg)/N <= m_inter/N.

    Differences from the original generator (src/cogent_empirical_validation.py):
      * the intra-class misconnection is rounded DOWN to the nearest feasible
        value (an odd N forces an even intra-class degree).  The original rounds
        it up, so with N = 25 only p_max in {0.04, 0.12, 0.20, 0.28, ...} was
        reachable; here p_max = m/N for every m >= 1;
      * with `rng`, the node order is permuted at random inside every class and,
        independently, for every pair of classes before the blocks are laid out.
        The graph stays regular (p_max is still exact), but it is no longer the
        same banded graph for every seed.  Without this, the graph-only accuracy
        was either 1 or ~0.15 (the deterministic graph is either recoverable or
        not, for every seed alike)."""
    classes = np.unique(labels)
    n = len(labels)
    A = np.zeros((n, n))
    idx = {c: np.flatnonzero(labels == c) for c in classes}
    N = len(idx[classes[0]])
    m_inter = int(min(max(round(p_target * N), 0), N))
    # p_ii = (N - intra_deg)/N <= m_inter/N, with N*intra_deg even
    intra_deg = N - max(m_inter, 1)
    if (N * intra_deg) % 2:
        intra_deg += 1
    intra_deg = min(intra_deg, N - 1)
    perm = (lambda v: rng.permutation(v)) if rng is not None else (lambda v: v)
    for c in classes:
        _add_circulant(A, perm(idx[c]), intra_deg)
    for i, ci in enumerate(classes):
        for cj in classes[i + 1 :]:
            _add_bipartite(A, perm(idx[ci]), perm(idx[cj]), m_inter)
    np.fill_diagonal(A, 0.0)
    return A


def empirical_pmax(A, labels):
    """
    p_max of a realized graph A according to paper eq. (6):

        n_bar_{v,j}  = (n_i - n_{v,i})  if j = i,    else  n_{v,j}
        N_bar_{i,j}  = sum_{v in C_i} n_bar_{v,j}
        bar_rho_{ij} = N_bar_{i,j} / (n_i n_j)
        delta_{ij}   = max_{v in C_i} | bar_rho_{ij} - n_bar_{v,j} / n_j |
        p_{ij}       = bar_rho_{ij} + 2 * delta_{ij}
        p_max        = max_{i,j} p_{ij}
    """
    classes = np.unique(labels)
    Kk = len(classes)
    n_v_j = np.zeros((len(labels), Kk))  # n_{v, C_j}
    for jj, cj in enumerate(classes):
        mask = labels == cj
        n_v_j[:, jj] = A[:, mask].sum(axis=1)

    pmax = 0.0
    for ii, ci in enumerate(classes):
        v_in_ci = np.where(labels == ci)[0]
        n_i = len(v_in_ci)
        for jj, cj in enumerate(classes):
            n_j = (labels == cj).sum()
            if ii == jj:
                nbar_v = n_i - n_v_j[v_in_ci, jj]  # missing intra
            else:
                nbar_v = n_v_j[v_in_ci, jj]  # spurious inter
            Nbar_ij = nbar_v.sum()
            bar_rho_ij = Nbar_ij / (n_i * n_j)
            delta_ij = float(np.max(np.abs(bar_rho_ij - nbar_v / n_j)))
            p_ij = bar_rho_ij + 2.0 * delta_ij
            pmax = max(pmax, float(p_ij))
    return pmax
