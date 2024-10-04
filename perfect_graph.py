import numpy as np

from graph import Graph


class PerfectGraph(Graph):
    def __init__(self, block_sizes) -> None:
        super().__init__()

        self.block_sizes = block_sizes
        self.A, self.A_bar, self.eigenvalues, self.E = self.create_adjacancy__matrix(
            block_sizes
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
