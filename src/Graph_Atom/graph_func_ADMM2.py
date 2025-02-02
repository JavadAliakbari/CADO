from src.Graph_Atom.graph_func import GraphFunc
from src.Graph_Atom.graph_sig import GraphSig
from src.utils.graph import Graph
from src.utils.utils import *


class GraphFuncADMM(GraphFunc):
    def __init__(
        self,
        graph: Graph,
    ) -> None:
        self.graph: Graph = graph
        self.Lambda = None
        self.Lp = None

    def __call__(self, signal: GraphSig):
        if signal is None:
            # term5
            if self.Lp is not None and self.Lambda is not None:
                term_5 = -matrix_dot_product(self.Lambda, self.Lp)
            else:
                term_5 = 0

            if self.Lp is not None:
                term_6 = matrix_dot_product(self.Lp, self.Lp)
            else:
                term_6 = 0

            return term_5 + config.beta / 2 * term_6

        term_1 = matrix_dot_product(self.graph.Abar, signal.L)
        # r = self.graph.A - signal.L
        # term_1 = 0.5 * matrix_dot_product(r, r)

        # term2
        if signal.Rv_inv is None:
            signal.calc_Rv_inv()
        # tt = np.einsum("ij,ijk->ik", self.graph.X, signal.Rv_inv)
        # term_2 = matrix_dot_product(tt, self.graph.X)
        # term_2 = np.trace(self.graph.cov @ signal.Rv_inv)
        term_2 = np.einsum("ijk,ijk->", self.graph.cov, signal.Rv_inv)

        # term3
        term_3 = np.trace(np.sum(signal.Rv, axis=0))

        # term4
        n = signal.pi.shape[0]
        term_4 = np.sum(signal.pi[np.arange(n), self.graph.y])

        # term5
        if self.Lp is not None:
            d = signal.pi - self.Lp
            term_5 = matrix_dot_product(self.Lambda, d)
        else:
            term_5 = 0

        # term6
        if self.Lp is not None and self.Lambda is not None:
            term_6 = matrix_dot_product(d, d)
        else:
            term_6 = 0

        obj = (
            -term_1
            + config.mu1 * term_2
            + config.mu1 * term_3
            - config.mu2 * term_4
            + term_5
            + config.beta / 2 * term_6
        )

        return obj

    def calc_grad(self, signal) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal)
        grad_R = self.calc_grad_R(signal)
        grad_pi = self.calc_grad_pi(signal)

        grad = GraphSig(grad_L, grad_R, grad_pi, add_eps=False)
        return grad

    def calc_grad_L(self, signal: GraphSig):
        return -self.graph.Abar

    def calc_grad_pi(self, signal: GraphSig):
        # Third block: Gradient with respect to pi_kv
        gradient_pi = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        # Create a one-hot encoded matrix
        train_labels = self.graph.y[self.graph.train_mask]
        train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
        gradient_pi[train_nodes, train_labels] = 1

        return (
            -config.mu2 * gradient_pi
            - self.Lambda
            - config.beta * (signal.pi - self.Lp)
        )
