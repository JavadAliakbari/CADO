import numpy as np
import networkx as nx

from graph import Graph


class SBM(Graph):
    def __init__(self, block_sizes, probs) -> None:
        super().__init__()
        self.block_sizes = block_sizes
        self.probs = probs

    def create_graph(self):
        g = nx.stochastic_block_model(self.block_sizes, self.probs)
        A = nx.adjacency_matrix(g)
        return A, Abar, eigen_values, eigen_vectors
