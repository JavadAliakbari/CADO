import numpy as np
from data import Data


class Graph(Data):
    def __init__(self) -> None:
        self.A = None
        self.edge_index = None
        self.Abar = None
        self.num_nodes = None
        self.X = None
        self.y = None
        self.num_features = None
        self.num_classes = None
        self.unique_labels = np.unique(self.y)

    # def __init__(self, edge_index, edge_weight, num_nodes) -> None:
    #     super().__init__()
    #     self.edge_index = edge_index
    #     self.edge_weight = edge_weight
    #     self.num_nodes = num_nodes


if __name__ == "__main__":
    graph = Graph()
    print(graph.Abar)
