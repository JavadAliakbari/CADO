import numpy as np

from data import Data


class Graph(Data):
    def __init__(
        self,
        num_nodes,
        train_ratio=1,
    ) -> None:
        self.A = None
        self.edge_index = None
        self.Abar = None
        self.num_nodes = num_nodes
        self.X = None
        self.y = None
        self.num_features = None
        self.num_classes = None
        self.unique_labels = np.unique(self.y)
        self.train_mask = None
        self.test_mask = None
        self.add_masks(train_ratio)

    def get_masks(self):
        return (self.train_mask, self.test_mask)

    def set_masks(self, masks):
        self.train_mask = masks[0]
        self.test_mask = masks[2]

    def add_masks(self, train_ratio=0.5):
        num_nodes = self.num_nodes
        self.train_mask, self.test_mask = Graph.split_indices(num_nodes, train_ratio)

    def split_indices(n, train_ratio=0.5):
        train_size = int(train_ratio * n)

        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_indices] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_indices] = True

        # np.equal(indices[:,None], train_indices).any(1)
        # train_mask = np.isin(indices, train_indices)
        # test_mask = np.isin(indices, test_indices)

        return train_mask, test_mask


if __name__ == "__main__":
    graph = Graph()
    print(graph.Abar)
