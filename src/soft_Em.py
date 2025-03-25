import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from test_Rv2 import create_features
from subspace_clustring import generate_cluster_data, visualize_high_dim_data
from utils.algorithm_utils_orig import *
from utils.utils import config

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


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


class softEM:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        # self.num_signal_features = num_signal_features

    def create_random_R(self, n, d):
        atoms = []
        coefs = np.random.rand(n, self.num_clusters)

        for _ in range(self.num_clusters):
            R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
                d, 0, config.rho_max - config.rho_min
            )
            atoms.append(R)

        coefs = project_to_simplex(coefs, 1)
        atoms = np.stack(atoms)
        return atoms, coefs

    def log_gaussian_density(x, R):
        """
        Computes the log of the (unnormalized) Gaussian density for x ~ N(0, R)
        up to an additive constant.

        Returns:
        log_det_term = -0.5 * log(det(R))
        quad_term    = -0.5 * x^T R^{-1} x
        and their sum.
        """
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite.")
        Rinv = np.linalg.inv(R)
        quad = -0.5 * (x.T @ Rinv @ x)
        log_det_term = -0.5 * logdet
        return log_det_term + quad

    def update_responsibilities(X, atoms, coefs):
        """
        E-step: Update the responsibilities (soft assignments) r_{iv} for each data point x_v.

        Parameters:
        X         : Data matrix (n_samples x n_features)
        U_list    : List of current subspace bases for clusters (each of shape n_features x d)
        w         : Array of cluster priors (length k)
        sigma_noise, sigma_signal: scalars
        Returns:
        r : n_samples x k array of responsibilities.
        """
        n_samples = X.shape[0]
        k = atoms.shape[0]
        log_resp = np.zeros((n_samples, k))

        for i in range(k):
            R_i = atoms[i]
            # R_i = atoms[i] + config.rho_min**2 * np.eye(atoms.shape[1])
            # Compute log density for each data point for cluster i.
            for v in range(n_samples):
                # log(w_i * N(x;0,R_i)) up to constant terms:
                log_resp[v, i] = np.log(
                    np.mean(coefs[:, i])
                ) + softEM.log_gaussian_density(X[v], R_i)

        # Normalize to get responsibilities
        # Use log-sum-exp trick for numerical stability.
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_resp_norm = log_resp - max_log
        resp = np.exp(log_resp_norm)
        resp = resp / np.sum(resp, axis=1, keepdims=True)
        return resp

    def update_subspaces(X, coefs, atoms):
        """
        M-step: For each cluster, update the subspace by computing the weighted covariance
        and then taking its top d eigenvectors.

        Parameters:
        X : Data matrix (n_samples x n_features)
        r : Responsibilities (n_samples x k)
        d : Subspace dimension (d > 1)
        Returns:
        U_list_new : list of updated subspace bases (each n_features x d)
        w_new      : updated cluster priors (array of length k)
        """
        k = coefs.shape[1]
        sigma_signal = np.sqrt(config.rho_max - config.rho_min)
        sigma_noise = np.sqrt(config.rho_min)
        th = (sigma_signal + sigma_noise) / 2

        for i in range(k):
            # Compute weighted covariance S_i = (sum_v r_{iv} x_v x_v^T) / (sum_v r_{iv})
            weights = coefs[:, i]
            # Weight for cluster i: average responsibility
            total_weight = np.sum(weights) + 1e-10  # avoid division by zero
            # Note: one could center X here if needed, but we assume zero mean.
            S_i = (X.T * weights) @ X / total_weight
            # Compute eigen-decomposition. eigh returns eigenvalues in ascending order.
            eigenvals, eigenvecs = np.linalg.eigh(S_i)
            # Take the top d eigenvectors (largest eigenvalues).
            mask = eigenvals > th
            eigenvals[mask] = sigma_signal**2
            eigenvals[~mask] = 0
            # eigenvals[:-2] = 0
            # eigenvals[-2:] = sigma_signal**2
            R_i = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            atoms[i] = R_i  # Update the atom

        return atoms

    def fit(self, X, total_iters):
        # Initialization
        n, num_features = X.shape
        atoms, coefs = self.create_random_R(n, num_features)

        for _ in range(total_iters):
            coefs = softEM.update_responsibilities(X, atoms, coefs)

            # M-step: update subspaces and cluster priors.
            atoms = softEM.update_subspaces(X, coefs, atoms)

        return atoms, coefs


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
    model = softEM(3)
    atoms, coefs = model.fit(X, 1000)

    R = calc_R(atoms, coefs)
    f_R = ff(R)

    y_pred = np.argmax(coefs, axis=1)
    ari = adjusted_rand_score(y_pred, clusters)

    Q_pred = np.array([np.linalg.eigh(R)[1] for R in atoms])
    visualize_high_dim_data(X, y_pred, Q)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}pred_clusters.png")
