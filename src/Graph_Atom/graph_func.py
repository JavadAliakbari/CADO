from Graph_Atom.graph_sig import GraphSig
from Atom.function import f
from utils.graph import Graph
from utils.utils import *


class GraphFunc(f):
    def __init__(self, graph: Graph, mu1=config.mu1, mu2=config.mu2) -> None:
        self.graph: Graph = graph
        self.mu1 = mu1
        self.mu2 = mu2

    def __call__(self, signal: GraphSig, components=False):
        if signal is None:
            return 0

        L_term = self.calc_L(signal)
        R_term = self.calc_R(signal)
        pi_term = self.calc_pi(signal)

        obj = L_term + R_term + pi_term

        if components:
            return obj, L_term, R_term, pi_term
        else:
            return obj

    def calc_L(self, signal: GraphSig):
        normalizer = signal.n if config.normalize else 1
        if config.Abar:

            L_term = -matrix_dot_product(self.graph.Abar, signal.L) / normalizer
        else:
            r = self.graph.A - signal.L
            L_term = 0.5 * matrix_dot_product(r, r) / normalizer

        return L_term

    def calc_R(self, signal: GraphSig):
        # term2
        normalizer = signal.d if config.normalize else 1
        if self.mu1 != 0:
            if signal.Rv_inv is None:
                signal.calc_Rv_inv()
            # tt = np.einsum("ij,ijk->ik", self.graph.X, signal.Rv_inv)
            # term_2 = matrix_dot_product(tt, self.graph.X)
            # term_2 = np.trace(self.graph.cov @ signal.Rv_inv)
            term_2 = np.einsum("ijk,ijk->", self.graph.cov, signal.Rv_inv) / normalizer

            # term3
            # term_3 = np.sum(np.log(np.linalg.det(signal.Rv))) / normalizer
            term_3 = np.trace(np.sum(signal.Rv, axis=0)) / normalizer
        else:
            term_2 = 0
            term_3 = 0

        return self.mu1 * term_2 + self.mu1 * term_3

    def calc_pi(self, signal: GraphSig):
        # term4
        normalizer = signal.c if config.normalize else 1
        if self.mu2 != 0:
            train_labels = self.graph.y[self.graph.train_mask]
            train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
            term_4 = np.sum(signal.pi[train_nodes, train_labels]) / normalizer
        else:
            term_4 = 0

        return -self.mu2 * term_4

    def calc_cov(self):
        return self.graph.cov

    def calc_grad(self, signal) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal)
        normalizer = signal.c * config.train_ratio if config.normalize else 1
        grad_R = self.calc_grad_R(signal) / normalizer
        grad_pi = self.calc_grad_pi(signal) / normalizer
        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig):
        normalizer = signal.n * config.p if config.normalize else 1

        if config.Abar:
            return (-self.graph.Abar) / normalizer
        else:
            return (-self.graph.A + signal.L) / normalizer

    def calc_grad_R(self, signal: GraphSig):
        # Second block: Gradient with respect to Rv_matrices
        if signal.Rv_inv is None:
            signal.calc_Rv_inv()

        gradient_Rv = signal.Rv_inv - np.einsum(
            "ijk,ikm,imn->ijn", signal.Rv_inv, self.graph.cov, signal.Rv_inv
        )

        return self.mu1 * gradient_Rv

    def calc_grad_pi(self, signal):
        # Third block: Gradient with respect to pi_kv
        gradient_pi = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        # Create a one-hot encoded matrix
        train_labels = self.graph.y[self.graph.train_mask]
        train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
        gradient_pi[train_nodes, train_labels] = 1

        return -self.mu2 * gradient_pi
