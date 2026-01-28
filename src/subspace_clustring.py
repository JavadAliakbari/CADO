import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def generate_cluster_data(
    n_samples, n_features, subspace_dim, sigma_signal, sigma_noise, center=None
):
    """
    Generate synthetic data for one cluster.

    Parameters:
      n_samples: number of points
      n_features: ambient dimension
      subspace_dim: dimension of the subspace
      sigma_signal: standard deviation (power) of the signal along the subspace
      sigma_noise: standard deviation (power) of the noise in the ambient space
      center: optional cluster center (if None, center is zero)

    Returns:
      data: n_samples x n_features array
      U: the orthonormal basis for the subspace (n_features x subspace_dim)
    """
    # If center is not given, use zero.
    if center is None:
        center = np.zeros(n_features)
    # Generate a random subspace basis (orthonormal)
    U, _ = np.linalg.qr(np.random.randn(n_features, subspace_dim))

    # Generate the signal component in the low-dimensional space.
    # Each coordinate is drawn from N(0, sigma_signal^2)
    signal = sigma_signal * np.random.randn(n_samples, subspace_dim)
    data_signal = signal @ U.T  # Project back to ambient space.

    # Generate additive noise in the ambient space.
    noise = sigma_noise * np.random.randn(n_samples, n_features)

    # Data points are the sum of the center, signal component, and noise.
    data = center + data_signal + noise
    return data, U


def compute_subspace(data, subspace_dim):
    """
    Compute a subspace basis and the center of the given data using SVD.
    """
    center = np.mean(data, axis=0)
    centered = data - center
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:subspace_dim].T  # top 'subspace_dim' directions
    return basis, center


def distance_to_subspace(point, basis, center):
    """
    Compute the distance from a point to its projection on the subspace.
    """
    diff = point - center
    projection = basis @ (basis.T @ diff)
    error = np.linalg.norm(diff - projection)
    return error


def k_subspaces(X, k, subspace_dim, n_iter=100):
    """
    Cluster data points into k clusters, each approximated by a subspace.
    """
    n_samples, n_features = X.shape
    clusters = np.random.randint(0, k, n_samples)

    subspaces = []
    centers = []
    for i in range(k):
        points = X[clusters == i]
        if points.shape[0] < subspace_dim:
            basis = np.random.randn(n_features, subspace_dim)
            basis, _ = np.linalg.qr(basis)  # Orthonormalize
            center = np.zeros(n_features)
        else:
            basis, center = compute_subspace(points, subspace_dim)
        subspaces.append(basis)
        centers.append(center)

    for iteration in range(n_iter):
        new_clusters = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            errors = []
            for j in range(k):
                err = distance_to_subspace(X[i], subspaces[j], centers[j])
                errors.append(err)
            new_clusters[i] = np.argmin(errors)

        if np.all(new_clusters == clusters):
            print(f"Converged at iteration {iteration}")
            break

        clusters = new_clusters.copy()
        for j in range(k):
            points = X[clusters == j]
            if points.shape[0] >= subspace_dim:
                basis, center = compute_subspace(points, subspace_dim)
                subspaces[j] = basis
                centers[j] = center

    return clusters, subspaces, centers


def visualize_high_dim_data(X, clusters, subspaces, centers=None):
    centers = centers or [np.zeros(X.shape[1]) for _ in range(len(subspaces))]
    """
    Visualize data (possibly high-dimensional) by projecting to 3 dimensions using PCA.
    Also projects the subspaces.
    """
    # Project data into 3 dimensions using PCA.
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    # Project centers into 3D.
    # centers_3d = [pca.transform(center.reshape(1, -1))[0] for center in centers]

    # For each subspace (assuming subspace_dim == 1 for a line),
    # generate points along the subspace in the original space and project them.
    k = len(centers)
    t_vals = np.linspace(-3, 3, 100)
    subspace_lines_3d = []
    for j in range(k):
        center = centers[j]
        basis = subspaces[j][:, 0]  # first basis vector for a line
        line_points = np.array([center + t * basis for t in t_vals])
        line_points_3d = pca.transform(line_points)
        subspace_lines_3d.append(line_points_3d)

    # Plot the projected data and subspace lines.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for j in range(k):
        idx = clusters == j
        ax.scatter(
            X_3d[idx, 0],
            X_3d[idx, 1],
            X_3d[idx, 2],
            color=colors[j % len(colors)],
            label=f"Cluster {j}",
            alpha=0.6,
        )
        line_points = subspace_lines_3d[j]
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            line_points[:, 2],
            color=colors[j % len(colors)],
            linewidth=3,
        )

    ax.set_title("Clusters and their projected subspaces (via PCA)")
    ax.legend()
    # plt.show()


# ------------------------------
# Example: Generate data from subspaces with defined signal and noise power, then cluster and visualize.
# ------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Set parameters.
    n_total = 1000  # total number of points
    n_features = 10  # ambient dimension
    k = 3  # number of clusters
    subspace_dim = 1  # dimension of each subspace (line)

    # Define signal and noise power for each cluster.
    # (You can also set different values per cluster if you want.)
    sigma_signal = 5.0  # signal standard deviation (power along the subspace)
    sigma_noise = 1.0  # noise standard deviation (ambient noise)

    # Generate clusters.
    X_list = []
    true_labels = []
    U_list = []  # to store the true subspace bases (for reference)
    samples_per_cluster = n_total // k
    true_centers = []
    true_subspaces = []
    for i in range(k):
        # Optionally, define a nonzero center for each cluster.
        center = np.zeros(n_features) * 2
        data, U = generate_cluster_data(
            samples_per_cluster,
            n_features,
            subspace_dim,
            sigma_signal,
            sigma_noise,
            center=center,
        )
        X_list.append(data)
        true_labels.extend([i] * samples_per_cluster)
        U_list.append(U)
        true_centers.append(center)
        true_subspaces.append(U)
    X = np.vstack(X_list)
    true_labels = np.array(true_labels)

    # Run the k-subspaces clustering algorithm.
    clusters, subspaces, centers = k_subspaces(X, k, subspace_dim, n_iter=100)

    # Measure ARI performance.
    ari = adjusted_rand_score(true_labels, clusters)
    print("Adjusted Rand Index (ARI):", ari)

    # Visualize using PCA to project to 3 dimensions.
    visualize_high_dim_data(X, true_labels, true_subspaces, true_centers)
    visualize_high_dim_data(X, clusters, subspaces, centers)

    plt.show()
