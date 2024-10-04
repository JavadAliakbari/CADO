from graph_grad import GraphGrad
from graph_sig import GraphSig
from function import f
from graph import Graph
from utils import *


class Graph_func(f):
    def __init__(self, signal: GraphSig, graph: Graph) -> None:
        self.signal: GraphSig = signal
        self.graph: Graph = graph

    def calc_objective(self):
        term_1 = matrix_dot_product(self.graph.Abar, self.signal.L)

        # term2
        Rv_matrices_inverse = np.array(
            [np.linalg.inv(matrix) for matrix in self.signal.Rv]
        )
        term_2 = matrix_dot_product(self.graph.X @ Rv_matrices_inverse, self.graph.X)

        # term3
        term_3 = np.trace(np.sum(self.signal.Rv, axis=0))

        # term4
        n = self.signal.pi.shape[0]
        term_4 = np.sum(self.signal.pi[np.arange(n), self.graph.y])
        obj = (
            -1 * term_1
            + config.mu1 * term_2
            + config.mu1 * term_3
            - config.mu2 * term_4
        )

        return obj

    def calc_grad(self) -> GraphGrad:
        grad_L = self.calc_grad_L()
        grad_R = self.calc_grad_R()
        grad_pi = self.calc_grad_pi()

        grad = GraphGrad(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self):
        return self.graph.Abar

    def calc_grad_R(self):
        # Second block: Gradient with respect to Rv_matrices
        Rv_matrices_inverse = np.array(
            [np.linalg.inv(matrix) for matrix in self.signal.Rv]
        )
        x_xtranspose = np.array([np.outer(xv, xv) for xv in self.graph.X])
        gradient_Rv = np.array(
            [
                -np.matmul(
                    np.matmul(Rv_matrices_inverse[i], x_xtranspose[i]),
                    Rv_matrices_inverse[i],
                )
                + np.eye(self.graph.X.shape[1])
                for i in range(Rv_matrices_inverse.shape[0])
            ]
        )

        return gradient_Rv

    def calc_grad_pi(self):
        # Third block: Gradient with respect to pi_kv
        tmp = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=int
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        tmp[np.arange(self.graph.num_nodes), self.graph.y] = (
            1  # Create a one-hot encoded matrix
        )
        gradient_pi = tmp
        return gradient_pi
