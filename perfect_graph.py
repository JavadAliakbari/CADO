import numpy as np

from graph import Graph
from utils import *


class PerfectGraph(Graph):
    def __init__(
        self,
        block_sizes,
        num_features,
        num_signal_features,
        train_ratio=1,
    ) -> None:
        num_nodes = sum(block_sizes)
        super().__init__(num_nodes, train_ratio)

        self.block_sizes = block_sizes
        self.num_features = num_features
        self.num_signal_features = num_signal_features
        self.num_classes = len(block_sizes)
        self.num_clusters = self.num_classes
        self.A, self.Abar, self.eigenvalues, self.E = self.create_adjacancy__matrix()
        self.b, self.pi, self.y = self.create_labels()

        self.Ri_matrices, self.Rv_matrices, self.X = self.create_features()

        self.cov = (
            np.einsum("ij,ik->ijk", self.X, self.X)
            - config.rho_min * np.eye(self.num_features)[None, :]
        )

    def create_adjacancy__matrix(self):
        """
        Create a block matrix with ones on the diagonal blocks and zeros elsewhere.

        Parameters:
        block_sizes (list of int): Sizes of the blocks along the diagonal.

        Returns:
        np.ndarray: The resulting block matrix.
        """

        K = len(self.block_sizes)
        # Calculate the total size of the matrix
        total_size = sum(self.block_sizes)
        # Initialize the full matrix with zeros
        block_matrix = np.zeros((total_size, total_size), dtype=int)

        # Fill the diagonal blocks with ones
        start_index = 0
        for size in self.block_sizes:
            block_matrix[
                start_index : start_index + size, start_index : start_index + size
            ] = 1
            start_index += size

        eigenvalues, eigenvectors = np.linalg.eigh(block_matrix)
        top_indices = np.argsort(eigenvalues)[
            -K:
        ]  # Indices of the K largest eigenvalues
        Lambda = eigenvalues[top_indices]  # The K largest eigenvalues
        E = eigenvectors[
            :, top_indices
        ]  # The eigenvectors corresponding to the largest eigenvalues, the dimension is nxK

        block_bar = block_matrix.copy()
        block_bar[block_bar == 0] = -1

        return block_matrix, block_bar, Lambda, E

    def create_features(self):
        ## Generate symmetric matrices Ri
        Ri_matrices = np.stack(
            [
                create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
                    self.num_features,
                    config.sig_rho_min,
                    config.sig_rho_max,
                    self.num_signal_features,
                )
                for _ in range(self.num_clusters)
            ],
            axis=0,
        )

        # Xi = [
        #     np.random.multivariate_normal(
        #         mean=np.zeros(self.num_features), cov=Ri, size=1
        #     ).flatten()
        #     for Ri in Ri_matrices
        # ]

        ## Calculate Rv based on the provided expression
        Rv_matrices = np.einsum(
            "ij,jkn->ikn", ((self.E**2) @ np.diag(self.eigenvalues)), Ri_matrices
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
        # norm_x = np.linalg.norm(X, axis=1)
        # X = np.diag(1/ norm_x) @ X
        # X = np.array([Xi[y] for y in self.y])

        # colors = ["r", "b", "g"]
        # # X_embed = TSNE(n_components=2, perplexity=30).fit_transform(X)
        # pca = PCA(n_components=2)
        # for ii in range(self.num_classes):
        #     mask = self.y == ii
        #     X_ii = X[mask]
        #     X_embed = pca.fit_transform(X_ii)
        #     plt.scatter(X_embed[:, 0], X_embed[:, 1], c=colors[ii])
        # plt.show()

        Rv_matrices_inverse = []
        for matrix in Rv_matrices:
            matrix_inv = np.linalg.inv(matrix)
            Rv_matrices_inverse.append(matrix_inv)
        Rv_matrices_inverse = np.array(Rv_matrices_inverse)
        # Rh =
        z = np.einsum("ij,ikj->ik", X, Rv_matrices_inverse)
        eyes = np.einsum("i,jk->ijk", np.ones(z.shape[0]), np.eye(X.shape[1]))
        gradient_Rv = eyes - np.einsum("ij,ik->ijk", z, z)

        grad_R = np.einsum("ijk,il->ljk", gradient_Rv, self.E**2)
        S, U = np.linalg.eigh(grad_R[2])
        mask1 = S > 0
        S[mask1] = config.rho_max
        S[~mask1] = config.rho_min
        R = U @ np.diag(S) @ U.T

        return Ri_matrices, Rv_matrices, X

    def create_labels(self):
        ## Generate the probability vector matrix b and generate matrix pi based on the provided expression
        b = create_multiple_noisy_probability_vectors(
            self.num_clusters, self.num_classes, noise_level=0.00
        )
        pi = ((self.E**2) @ np.diag(self.eigenvalues)) @ b

        ## Labels y based on the probability distributions in pi
        y = np.array([np.random.choice(self.num_classes, p=pi_row) for pi_row in pi])

        return b, pi, y
