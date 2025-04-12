import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from subspace_clustring import (
    generate_cluster_data,
    visualize_high_dim_data2,
)
from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def create_features(num_nodes, num_features, num_signal_features, num_clusters):
    node_per_cluster = num_nodes // num_clusters
    Ris = []
    X_list = []
    clusters = []
    Q_list = []
    for i in range(num_clusters):
        Ri = create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
            num_features,
            config.rho_min,
            config.rho_max,
            num_signal_features,
        )

        eigenvals, eigenvecs = np.linalg.eigh(Ri)
        # Take the top d eigenvectors (largest eigenvalues).
        Q = eigenvecs[:, -num_signal_features:]

        X = np.random.multivariate_normal(
            mean=np.zeros(num_features), cov=Ri, size=node_per_cluster
        )

        Ris.append(Ri)
        X_list.append(X)
        Q_list.append(Q)
        clusters.extend([i] * node_per_cluster)

    Ri_matrices = np.array(Ris)
    X = np.concatenate(X_list)
    Q = np.array(Q_list)
    clusters = np.array(clusters)

    return Ri_matrices, X, Q, clusters


def create_features2(num_nodes, num_features, num_signal_features, num_clusters):
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


class softEM:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        # self.num_signal_features = num_signal_features

    def create_random_R(self, n, d):
        atoms = []
        # coefs = np.random.rand(n, self.num_clusters)
        coefs = np.ones((n, self.num_clusters)) / self.num_clusters

        th = (config.rho_max + config.rho_min) / 2
        for _ in range(self.num_clusters):
            # Draw a random n_features x d matrix and orthonormalize its columns.
            A = np.random.randn(d, 2)
            # Use QR decomposition to get an orthonormal basis.
            Q, _ = np.linalg.qr(A)
            R_i = config.rho_min * np.eye(d) + (config.rho_max - config.rho_min) * (
                Q @ Q.T
            )

            atoms.append(R_i)

        # coefs = project_to_simplex(coefs, 1)
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
        th = 10 * config.rho_min
        d = X.shape[1]

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
            th = eigenvals[-2]
            mask = eigenvals >= th
            eigenvals[mask] = config.rho_max - config.rho_min
            eigenvals[~mask] = 0
            # eigenvals[:-2] = 0
            # eigenvals[-2:] = sigma_signal**2
            R_i = eigenvecs @ np.diag(
                eigenvals
            ) @ eigenvecs.T + config.rho_min * np.eye(d)
            atoms[i] = R_i  # Update the atom

        return atoms

    def fit(self, X, total_iters, tol=1e-5):
        # Initialization
        n, num_features = X.shape
        atoms, coefs = self.create_random_R(n, num_features)

        bar = tqdm(total=total_iters)
        prev_ll = -np.inf
        for iteration in range(total_iters):
            coefs = softEM.update_responsibilities(X, atoms, coefs)

            # M-step: update subspaces and cluster priors.
            atoms = softEM.update_subspaces(X, coefs, atoms)

            y_pred = np.argmax(coefs, axis=1)
            ari = adjusted_rand_score(y_pred, clusters)

            # (Optional) Compute the complete-data log–likelihood for monitoring.
            ll = 0
            for coef, R_i in zip(coefs.T, atoms):
                sign, logdet = np.linalg.slogdet(R_i)
                if sign <= 0:
                    raise ValueError("Non–positive definite covariance encountered.")
                # The constant term from the Gaussian density is omitted.
                tmp = np.mean(coef) * np.exp(
                    -0.5
                    * (np.einsum("nj, jk, nk->n", X, np.linalg.inv(R_i), X) + logdet)
                )
                # To avoid log(0) we add a small constant.
                ll += np.log(np.sum(tmp) + 1e-10)

            # Check convergence (using change in log-likelihood).
            if iteration > 0 and np.abs(ll - prev_ll) < tol:
                print(f"Converged at iteration {iteration}, log-likelihood: {ll:.5f}")
                break
            prev_ll = ll
            bar.set_postfix_str(f"(ARI): {ari:0.4f}, log-likelihood: {ll:.4f}")
            # print(f"Iteration {iteration}, ")
            bar.update()

        return atoms, coefs


if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [
        2000,
        2000,
        2000,
    ]  # Number of training samples in each cluster
    num_features = 10  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    config.tau = n
    config.mu0 = 5
    c = len(samples_per_cluster)

    # probs = np.eye(c)
    Ri_matrices, X, Q, clusters = create_features(
        n, num_features, num_signal_features, c
    )
    visualize_high_dim_data2(X, clusters, Q)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}true_clusters.png")
    # plt.show()

    R_opt = Ri_matrices[clusters]

    ## Initialization
    model = softEM(3)
    atoms, coefs = model.fit(X, 100)

    y_pred = np.argmax(coefs, axis=1)
    ari = adjusted_rand_score(y_pred, clusters)
    print("Adjusted Rand Index (ARI):", ari)

    Q_pred = np.array([np.linalg.eigh(R)[1] for R in atoms])
    visualize_high_dim_data2(X, y_pred, Q_pred)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}pred_clusters.png")
    plt.show()
