from graph_sig import GraphSig
from function import f
from graph import Graph
from utils import *


class GraphFunc(f):
    def __init__(
        self,
        graph: Graph,
    ) -> None:
        self.graph: Graph = graph

    def __call__(self, signal: GraphSig):
        if signal is None:
            return 0
        # term_1 = -matrix_dot_product(self.graph.A, signal.L)
        r = self.graph.A - signal.L
        term_1 = 0.5 * matrix_dot_product(r, r)

        # term2
        Rv_matrices_inverse = []
        for matrix in signal.Rv:
            try:
                matrix_inv = np.linalg.inv(matrix)
            except:
                matrix_inv = 0 * matrix

            Rv_matrices_inverse.append(matrix_inv)
        Rv_matrices_inverse = np.array(Rv_matrices_inverse)
        tt = np.einsum("ij,ijk->ik", self.graph.X, Rv_matrices_inverse)
        term_2 = matrix_dot_product(tt, self.graph.X)

        # term3
        term_3 = np.trace(np.sum(signal.Rv, axis=0))

        # term4
        n = signal.pi.shape[0]
        term_4 = np.sum(signal.pi[np.arange(n), self.graph.y])
        obj = term_1 + config.mu1 * term_2 + config.mu1 * term_3 - config.mu2 * term_4

        return obj

    def calc_grad(self, signal) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal)
        grad_R = self.calc_grad_R(signal)
        grad_pi = self.calc_grad_pi(signal)

        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig):
        return self.graph.A - signal.L
        # return self.graph.A

    def calc_grad_R(self, signal: GraphSig):
        # Second block: Gradient with respect to Rv_matrices

        Rv_matrices_inverse = []
        for matrix in signal.Rv:
            try:
                matrix_inv = np.linalg.inv(matrix)
            except:
                matrix_inv = 0 * matrix

            Rv_matrices_inverse.append(matrix_inv)
        Rv_matrices_inverse = np.array(Rv_matrices_inverse)
        z = np.einsum("ij,ikj->ik",self.graph.X, Rv_matrices_inverse)
        eyes = np.einsum(
            "i,jk->ijk", np.ones(z.shape[0]), np.eye(self.graph.X.shape[1])
        )
        gradient_Rv = eyes - np.einsum("ij,ik->ijk", z, z)

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
