import numpy as np
import networkx as nx
import scipy.sparse as sp

from utils.graph import Graph
from utils.utils import (
    create_Ri_symmetric_matrix_with_constrained_eigenvalues2,
    create_multiple_noisy_probability_vectors,
    data,
)


# Add scalar multiplication support for SparseAbar objects.
class SparseAbar:
    def __init__(self, A, scale=1):
        # Convert dense A to sparse CSR format.
        if not sp.issparse(A):
            self.A = sp.csr_matrix(A)
        else:
            self.A = A
        self.shape = self.A.shape
        self.scale = scale

    def __matmul__(self, v):
        # Compute: scale * (2*A.dot(v) - np.sum(v)*ones)
        res = 2 * self.scale * self.A.dot(v)
        res -= self.scale * np.sum(v) * np.ones(self.shape[0])
        return res

    def __rmatmul__(self, v):
        # Compute v @ Abar.
        res = 2 * self.scale * (v.dot(self.A))
        res -= self.scale * np.sum(v) * np.ones(self.shape[1])
        return res

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return SparseAbar(self.A, self.scale * c)
        return NotImplemented

    def __rmul__(self, c):
        return self.__mul__(c)

    def __truediv__(self, c):
        if isinstance(c, (int, float)):
            return SparseAbar(self.A, self.scale / c)
        return NotImplemented

    def __rtruediv__(self, c):
        return self.__truediv__(c)

    def __neg__(self):
        return SparseAbar(self.A, -self.scale)


class SBM(Graph):
    def __init__(
        self,
        block_sizes,
        probs,
        num_features,
        num_signal_features,
        train_ratio=1,
        rho_min=data.sig_rho_min,
        rho_max=data.sig_rho_max,
        gmm_center_min=None,
        gmm_sigma=None,
    ) -> None:
        num_nodes = sum(block_sizes)
        super().__init__(num_nodes, train_ratio)
        self.block_sizes = block_sizes
        self.probs = probs
        self.num_features = num_features
        self.num_signal_features = num_signal_features
        self.num_classes = len(block_sizes)
        self.num_clusters = self.num_classes
        self.A, self.Abar, self.λ_SBM, self.E_SBM = self.create_adjacancy__matrix()
        self.L_opt, _, self.λ, self.E = self.create_partionioned_graph()
        self.b, self.pi, self.y = self.create_labels()
        self.gmm_center_min = gmm_center_min if gmm_center_min is not None else 2.0
        self.gmm_sigma = gmm_sigma if gmm_sigma is not None else 1.0

        self.Ri_matrices, self.Rv_matrices, self.X, self.cluster_centers = (
            self.create_features_gmm()
        )

        self.X /= np.linalg.norm(self.X, axis=1)[:, None]
        self.cov = (
            np.einsum("ij,ik->ijk", self.X, self.X)
            # - data.rho_min * np.eye(self.num_features)[None, :]
        )

        self.train_labels = self.y[self.train_mask]
        self.train_nodes = np.arange(self.num_nodes)[self.train_mask]

    def create_partionioned_graph(self):
        """
        Create a block matrix with ones on the diagonal blocks and zeros elsewhere.

        Parameters:
        block_sizes (list of int): Sizes of the blocks along the diagonal.

        Returns:
        np.ndarray: The resulting block matrix.
        """

        K = len(self.block_sizes)
        # Calculate the total size of the matrix
        # Initialize the full matrix with zeros
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        # Fill the diagonal blocks with ones
        start_index = 0
        for size in self.block_sizes:
            A[start_index : start_index + size, start_index : start_index + size] = 1
            start_index += size

        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_indices = np.argsort(eigenvalues)[
            -K:
        ]  # Indices of the K largest eigenvalues
        Lambda = eigenvalues[top_indices]  # The K largest eigenvalues
        E = eigenvectors[
            :, top_indices
        ]  # The eigenvectors corresponding to the largest eigenvalues, the dimension is nxK

        Abar = A.copy()
        Abar[Abar == 0] = -1

        return A, Abar, Lambda, E

    def create_adjacancy__matrix(self):
        K = len(self.block_sizes)
        n = sum(self.block_sizes)

        g = nx.stochastic_block_model(self.block_sizes, self.probs)
        A = nx.adjacency_matrix(g).toarray()
        np.fill_diagonal(A, 1)

        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
        # normalized_adj = D_inv_sqrt @ A @ D_inv_sqrt

        Abar = SparseAbar(A)

        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_indices = np.argsort(eigenvalues)[-K:]
        Lambda = eigenvalues[top_indices]
        E = eigenvectors[:, top_indices]

        return A, Abar, Lambda, E

    def create_features(self):
        ## Generate symmetric matrices Ri
        Ri_matrices = np.stack(
            [
                create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
                    self.num_features,
                    self.rho_min,
                    self.rho_max,
                    self.num_signal_features,
                )
                for _ in range(self.num_clusters)
            ],
            axis=0,
        )

        ## Calculate Rv based on the provided expression
        Rv_matrices = (
            np.einsum("ij,jkn->ikn", ((self.E**2) @ np.diag(self.λ)), Ri_matrices)
            # + data.rho_min * np.eye(self.num_features)[None, :]
        )

        # Generate feature vectors X based on the multivariate normal distribution
        X = np.array(
            [
                np.random.multivariate_normal(
                    mean=np.zeros(self.num_features), cov=Rv, size=1
                ).flatten()
                for Rv in Rv_matrices
            ]
        )

        return Ri_matrices, Rv_matrices, X

    def create_labels(self):
        ## Generate the probability vector matrix b and generate matrix pi based on the provided expression
        b = create_multiple_noisy_probability_vectors(
            self.num_clusters, self.num_classes, noise_level=0.00
        )

        # pi = []
        # for i, t in enumerate(self.block_sizes):
        #     pi.append(np.repeat(b[i, :][None, :], t, axis=0))
        # pi = np.vstack(pi)
        pi = ((self.E**2) @ np.diag(self.λ)) @ b
        # # pi = np.round(pi, 1)
        pi /= np.sum(pi, axis=1)[:, None]

        ## Labels y based on the probability distributions in pi
        y = np.array([np.random.choice(self.num_classes, p=pi_row) for pi_row in pi])

        return b, pi, y

    def create_features_gmm(self):
        """
        Generate features using Gaussian Mixture Model.
        Each cluster has a mean vector with values >= gmm_center_min and fixed covariance gmm_sigma^2 * I.

        Returns:
            Ri_matrices: Covariance matrices for each cluster (num_clusters x num_features x num_features)
            Rv_matrices: Per-node covariance matrices (num_nodes x num_features x num_features)
            X: Feature matrix (num_nodes x num_features)
            cluster_centers: Mean vectors for each cluster (num_clusters x num_features)
        """
        # Generate cluster centers with values >= gmm_center_min
        cluster_centers = []
        for i in range(self.num_clusters):
            # Generate random center with values between gmm_center_min and gmm_center_min + random offset
            center = self.gmm_center_min + np.random.uniform(
                0, 2, size=self.num_features
            )
            cluster_centers.append(center)
        cluster_centers = np.array(cluster_centers)

        # Fixed covariance matrix for all clusters: sigma^2 * I
        cluster_covariance = (self.gmm_sigma**2) * np.eye(self.num_features)

        # Create Ri_matrices (same for all clusters in this simple GMM)
        Ri_matrices = np.stack(
            [cluster_covariance for _ in range(self.num_clusters)], axis=0
        )

        # Generate features for each node based on cluster assignments
        X = []
        Rv_list = []

        start_idx = 0
        for cluster_idx, block_size in enumerate(self.block_sizes):
            # Generate samples from this cluster's Gaussian
            cluster_samples = np.random.multivariate_normal(
                mean=cluster_centers[cluster_idx],
                cov=cluster_covariance,
                size=block_size,
            )
            X.append(cluster_samples)

            # Rv for this cluster (same covariance for all nodes in cluster)
            Rv_cluster = np.stack(
                [cluster_covariance for _ in range(block_size)], axis=0
            )
            Rv_list.append(Rv_cluster)

            start_idx += block_size

        X = np.vstack(X)
        Rv_matrices = np.vstack(Rv_list)

        return Ri_matrices, Rv_matrices, X, cluster_centers
