import numpy as np

from graph import Graph
from utils import *


class PerfectGraph(Graph):
    def __init__(self, block_sizes, num_features) -> None:
        super().__init__()

        self.block_sizes = block_sizes
        self.num_features = num_features
        self.num_classes = len(block_sizes)
        self.num_clusters = self.num_classes
        self.num_nodes = sum(block_sizes)
        self.A, self.Abar, self.eigenvalues, self.E = self.create_adjacancy__matrix()

        self.Ri_matrices, self.Rv_matrices, self.X = self.create_features()
        self.b, self.pi, self.y = self.create_labels()

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
                create_Ri_symmetric_matrix_with_constrained_eigenvalues(
                    self.num_features,
                    config.rho_min,
                    config.rho_max,
                    noise_level=config.noise_level_Ri,
                    include_extremes=config.include_rhos,
                )
                for _ in range(self.num_clusters)
            ],
            axis=0,
        )

        ## Calculate Rv based on the provided expression
        Rv_matrices = np.einsum(
            "ij,jkn->ikn", ((self.E**2) @ np.diag(self.eigenvalues)), Ri_matrices
        )

        ## Generate feature vectors X based on the multivariate normal distribution
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
        pi = ((self.E**2) @ np.diag(self.eigenvalues)) @ b

        ## Labels y based on the probability distributions in pi
        y = np.array([np.random.choice(self.num_classes, p=pi_row) for pi_row in pi])

        return b, pi, y
