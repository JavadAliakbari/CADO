import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from Graph_Atom.graph_func import GraphFunc

from subspace_clustring import generate_cluster_data, visualize_high_dim_data
from utils.algorithm_utils_orig import *
from utils.utils import config

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def create_features(num_nodes, num_features, num_signal_features, num_clusters):
    sigma_signal = np.sqrt(config.sig_rho_max - config.sig_rho_min)
    sigma_noise = np.sqrt(config.sig_rho_min)
    node_per_cluster = num_nodes // num_clusters
    Ris = []
    # Rvs = []
    X_list = []
    clusters = []
    Q_list = []
    for i in range(num_clusters):
        X, Q = generate_cluster_data(
            node_per_cluster,
            num_features,
            num_signal_features,
            sigma_signal,
            sigma_noise,
        )
        Ri = sigma_signal**2 * np.einsum("ij,kj->ik", Q, Q) + sigma_noise**2
        # Rv = np.einsum(
        #     "i,jk->ijk",
        #     np.ones(node_per_cluster),
        #     Ri + config.rho_min * np.eye(num_features),
        # )

        Ris.append(Ri)
        # Rvs.append(Rv)
        X_list.append(X)
        clusters.extend([i] * node_per_cluster)
        Q_list.append(Q)

    Ri_matrices = np.array(Ris)
    # Rv_matrices = np.stack(Rvs)
    X = np.concatenate(X_list)
    Q = np.array(Q_list)
    clusters = np.array(clusters)

    return Ri_matrices, X, Q, clusters


def create_random_R(n, d, c, num_atoms):
    atoms = []
    coefs = np.random.rand(n, num_atoms)

    for _ in range(num_atoms):
        R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
            d, 0, config.rho_max - config.rho_min
        )
        atoms.append(R)

    coefs = project_to_simplex(coefs, 1)
    atoms = np.stack(atoms)
    return atoms, coefs


def proximal_conditional_descent(
    atoms: Atoms, coefs: Coefs, f: GraphFunc, num_atoms, iter=0
):
    """
    Proximal gradient descent algorithm
    """
    # atoms, coefs = create_random_atoms(sig.n, sig.d, sig.c, num_atoms)
    for itr in range(config.local_iter):
        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        for ind in idx:
            R = calc_R(atoms, coefs)  # Update signal
            grad_R = f.calc_grad(R)
            grad_Ri = np.einsum("ijk,i->jk", grad_R, coefs[:, ind])
            atoms[ind] -= 2 * config.lr * grad_Ri  # Update atom
            atoms[ind] = project_to_psd_cone(
                atoms[ind], 0, config.rho_max - config.rho_min
            )
        if iter % 10 == 0:
            coefs = update_coefs(atoms, f, coefs, 0.01, 1)

    return atoms, coefs


def update_coefs(atoms: Atoms, f: GraphFunc, coefs: Coefs, lr, itr):
    for _ in range(itr):
        R = calc_R(atoms, coefs)
        grad = f.calc_grad(R)
        g = np.einsum("njk,ijk->ni", grad, atoms)
        coefs = coefs - lr * g
        # coefs[coefs < 0] = 0
        coefs = project_to_simplex(coefs, 1)

    return coefs


class func:
    def __init__(self, X):
        self.X = X
        self.cov = np.einsum("ij,ik->ijk", self.X, self.X)
        self.d = self.X.shape[1]

    def __call__(self, R):
        R_inv = np.array([invert_matrix(matrix) for matrix in R])
        term_2 = np.einsum("ijk,ijk->", self.cov, R_inv)
        # term_3 = np.sum(np.log(np.linalg.det(R)))
        term_3 = np.trace(np.sum(R, axis=0))
        obj = term_2 + term_3

        return obj

    def calc_grad(self, R):
        R_inv = np.array([invert_matrix(matrix) for matrix in R])
        grad = np.eye(self.d)[None, :] - np.einsum(
            "ijk,ikm,imn->ijn", R_inv, self.cov, R_inv
        )
        return grad


def calc_R(atoms, coefs):
    d = atoms.shape[1]
    return (
        np.einsum("ni,ijk->njk", coefs, atoms) + config.rho_min * np.eye(d, d)[None, :]
    )


if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [
        600,
        600,
        600,
    ]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    config.tau = n
    config.mu0 = 5
    c = len(samples_per_cluster)
    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
    # probs = np.eye(c)
    Ri_matrices, X, Q, clusters = create_features(
        n, num_features, num_signal_features, c
    )
    visualize_high_dim_data(X, clusters, Q)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}true_clusters.png")
    # plt.show()

    R_opt = Ri_matrices[clusters]

    ff = func(X)
    f_opt = ff(R_opt)

    ## Initialization
    atoms, coefs = create_random_R(n, num_features, c, c)
    num_atoms = len(atoms)

    R = calc_R(atoms, coefs)
    # sig = EmptySig(n, num_features, c)

    f_R = ff(R)

    bar = tqdm(total=config.iters)
    f_values = []
    accuracies = []

    for ii in range(config.iters):
        atoms, coefs = proximal_conditional_descent(
            atoms, coefs, ff, num_atoms, iter=ii
        )
        R = calc_R(atoms, coefs)
        f_R = ff(R)

        y_pred = np.argmax(coefs, axis=1)
        ari = adjusted_rand_score(y_pred, clusters)

        results = {
            f"f": f"{f_R:,.4f}",
            "acc": ari,
            "f_opt": f"{f_opt:,.4f}",
        }

        bar.set_postfix(results)
        bar.update()
        f_values.append(f_R)
        accuracies.append(ari)
