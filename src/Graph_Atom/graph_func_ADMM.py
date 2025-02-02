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

    def set_ADMM_params(self, Lambda, Lp):
        self.Lambda = Lambda
        self.Lp = Lp

    def __call__(self, signal: GraphSig):
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
            d = signal.L - self.Lp
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
        # return self.graph.A - signal.L
        # return self.graph.Abar - config.beta * (signal.L - Lp)
        return -self.graph.Abar + self.Lambda + config.beta * (signal.L - self.Lp)
