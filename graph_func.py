from graph_sig import GraphSig
from function import f
from graph import Graph
from utils import *


class GraphFunc(f):
    def __init__(
        self,
        graph: Graph,
        # signal: GraphSig = None,
    ) -> None:
        # self.signal: GraphSig = signal
        self.graph: Graph = graph

    # def update_sig(self, sig: GraphSig):
    #     self.signal = sig

    def __call__(self, signal: GraphSig):
        if signal is None:
            return 0
        # term_1 = matrix_dot_product(self.graph.Abar, signal.L)
        r = self.graph.A - signal.L
        term_1 = 0.5 * matrix_dot_product(r, r)

        # term2
        try:
            Rv_matrices_inverse = np.array(
                [np.linalg.inv(matrix) for matrix in signal.Rv]
            )
            term_2 = matrix_dot_product(
                self.graph.X @ Rv_matrices_inverse, self.graph.X
            )
        except:
            term_2 = 0

        # term3
        term_3 = np.trace(np.sum(signal.Rv, axis=0))

        # term4
        n = signal.pi.shape[0]
        term_4 = np.sum(signal.pi[np.arange(n), self.graph.y])
        obj = (
            -1 * term_1
            + config.mu1 * term_2
            + config.mu1 * term_3
            - config.mu2 * term_4
        )

        return obj

    def calc_grad(self, signal) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal)
        grad_R = self.calc_grad_R(signal)
        grad_pi = self.calc_grad_pi(signal)

        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal):
        r = self.graph.A - signal.L
        return r
        # return self.graph.Abar

    def calc_grad_R(self, signal: GraphSig):
        # Second block: Gradient with respect to Rv_matrices
        Rv_matrices_inverse = np.array([np.linalg.inv(matrix) for matrix in signal.Rv])
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

    def calc_grad_pi(self, signal):
        # Third block: Gradient with respect to pi_kv
        tmp = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        tmp[np.arange(self.graph.num_nodes), self.graph.y] = (
            1  # Create a one-hot encoded matrix
        )
        gradient_pi = tmp
        return gradient_pi
