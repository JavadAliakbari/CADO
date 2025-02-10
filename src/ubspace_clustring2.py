from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import slogdet, inv, eigh, norm
from sklearn.metrics import adjusted_rand_score

from subspace_clustring import visualize_high_dim_data


def compute_R(U, sigma_noise, sigma_signal):
    """
    Given a subspace basis U (n_features x d), return the covariance:
         R = sigma_noise^2 I + sigma_signal^2 U U^T.
    """
    n_features = U.shape[0]
    return sigma_noise**2 * np.eye(n_features) + sigma_signal**2 * (U @ U.T)


def log_gaussian_density(x, R):
    """
    Computes the log of the (unnormalized) Gaussian density for x ~ N(0, R)
    up to an additive constant.

    Returns:
      log_det_term = -0.5 * log(det(R))
      quad_term    = -0.5 * x^T R^{-1} x
    and their sum.
    """
    sign, logdet = slogdet(R)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    Rinv = inv(R)
    quad = -0.5 * (x.T @ Rinv @ x)
    log_det_term = -0.5 * logdet
    return log_det_term + quad


def update_responsibilities(X, U_list, w, sigma_noise, sigma_signal):
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
    k = len(U_list)
    log_resp = np.zeros((n_samples, k))

    for i in range(k):
        R_i = compute_R(U_list[i], sigma_noise, sigma_signal)
        # Compute log density for each data point for cluster i.
        for v in range(n_samples):
            # log(w_i * N(x;0,R_i)) up to constant terms:
            log_resp[v, i] = np.log(w[i]) + log_gaussian_density(X[v], R_i)

    # Normalize to get responsibilities
    # Use log-sum-exp trick for numerical stability.
    max_log = np.max(log_resp, axis=1, keepdims=True)
    log_resp_norm = log_resp - max_log
    resp = np.exp(log_resp_norm)
    resp = resp / np.sum(resp, axis=1, keepdims=True)
    return resp


def update_subspaces(X, r, d):
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
    n_samples, n_features = X.shape
    k = r.shape[1]
    U_list_new = []
    w_new = np.zeros(k)

    for i in range(k):
        # Weight for cluster i: average responsibility
        w_new[i] = np.mean(r[:, i])
        # Compute weighted covariance S_i = (sum_v r_{iv} x_v x_v^T) / (sum_v r_{iv})
        weights = r[:, i]
        total_weight = np.sum(weights) + 1e-10  # avoid division by zero
        # Note: one could center X here if needed, but we assume zero mean.
        S_i = (X.T * weights) @ X / total_weight
        # Compute eigen-decomposition. eigh returns eigenvalues in ascending order.
        eigenvals, eigenvecs = eigh(S_i)
        # Take the top d eigenvectors (largest eigenvalues).
        U_i = eigenvecs[:, -d:]
        # Ensure the columns are orthonormal (they should be from eigh).
        U_list_new.append(U_i)

    return U_list_new, w_new


def mixture_subspace_em(X, k, sigma_noise, sigma_signal, d, max_iter=100, tol=1e-5):
    """
    EM-like algorithm to cluster data points and estimate higher-dimensional subspaces.

    The model for cluster i is:
         x ~ N(0, R_i),    with   R_i = sigma_noise^2 I + sigma_signal^2 U_i U_i^T,
    and we assume a mixture model
         p(x) = sum_i w_i N(0, R_i).

    The algorithm returns:
      - r      : n_samples x k array of soft assignments (responsibilities)
      - U_list : list of estimated subspace bases (each n_features x d)
      - w      : estimated cluster priors.
    """
    n_samples, n_features = X.shape

    # Initialize cluster priors uniformly.
    w = np.ones(k) / k

    # Initialize each cluster’s subspace with a random orthonormal basis.
    U_list = []
    for i in range(k):
        # Draw a random n_features x d matrix and orthonormalize its columns.
        A = np.random.randn(n_features, d)
        # Use QR decomposition to get an orthonormal basis.
        Q, _ = np.linalg.qr(A)
        U_list.append(Q[:, :d])

    prev_ll = -np.inf
    for iteration in range(max_iter):
        # E-step: update responsibilities.
        r = update_responsibilities(X, U_list, w, sigma_noise, sigma_signal)

        # M-step: update subspaces and cluster priors.
        U_list, w = update_subspaces(X, r, d)

        # (Optional) Compute the complete-data log–likelihood for monitoring.
        ll = 0
        for v in range(n_samples):
            tmp = 0
            for i in range(k):
                R_i = compute_R(U_list[i], sigma_noise, sigma_signal)
                sign, logdet = slogdet(R_i)
                if sign <= 0:
                    raise ValueError("Non–positive definite covariance encountered.")
                # The constant term from the Gaussian density is omitted.
                tmp += w[i] * np.exp(-0.5 * (np.dot(X[v], inv(R_i) @ X[v]) + logdet))
            # To avoid log(0) we add a small constant.
            ll += np.log(tmp + 1e-10)

        # Check convergence (using change in log-likelihood).
        if iteration > 0 and np.abs(ll - prev_ll) < tol:
            print(f"Converged at iteration {iteration}, log-likelihood: {ll:.5f}")
            break
        prev_ll = ll
        print(f"Iteration {iteration}, log-likelihood: {ll:.5f}")

    return r, U_list, w


# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    # Parameters for data generation.
    n_samples = 10000
    n_features = 10  # ambient dimension
    k = 3  # number of clusters
    d = 2  # subspace dimension (d > 1)

    sigma_signal = 5.0
    sigma_noise = .2

    # Generate synthetic data.
    # All clusters share the same mean (zero) and same sigma_noise and sigma_signal.
    # The only difference is in the subspace (i.e. the signal direction or subspace).
    X_list = []
    true_labels = []
    true_U = []  # for reference, the true subspace bases
    samples_per_cluster = n_samples // k
    for i in range(k):
        # Generate a random n_features x d subspace basis.
        A = np.random.randn(n_features, d)
        Q, _ = np.linalg.qr(A)
        U_true = Q[:, :d]
        true_U.append(U_true)

        # Generate the signal: low-dimensional coefficients.
        s = sigma_signal * np.random.randn(samples_per_cluster, d)
        signal = s @ U_true.T

        # Generate additive noise.
        noise = sigma_noise * np.random.randn(samples_per_cluster, n_features)
        X_cluster = signal + noise
        X_list.append(X_cluster)
        true_labels.extend([i] * samples_per_cluster)

    X = np.vstack(X_list)
    true_labels = np.array(true_labels)

    # Run the EM-like algorithm.
    r, U_list, w = mixture_subspace_em(
        X, k, sigma_noise, sigma_signal, d, max_iter=100, tol=1e-5
    )

    # The responsibilities r (of shape n_samples x k) give the soft assignments.
    # The estimated subspaces are in U_list and the cluster priors in w.
    print("Final cluster priors:", w)
    print("Final soft assignments (first 10 data points):")
    print(r[:10])

    clusters = np.argmax(r, axis=1)

    # Measure ARI performance.
    ari = adjusted_rand_score(true_labels, clusters)
    print("Adjusted Rand Index (ARI):", ari)

    # Visualize using PCA to project to 3 dimensions.
    visualize_high_dim_data(
        X, true_labels, true_U, np.zeros((k, n_features)), n_features
    )
    visualize_high_dim_data(X, clusters, U_list, np.zeros((k, n_features)), n_features)

    plt.show()
