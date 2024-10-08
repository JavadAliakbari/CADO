from graph_func import GraphFunc
from graph_sig import GraphSig
from graph import Graph
from utils import *


class GraphFuncADMM(GraphFunc):
    def __init__(
        self,
        graph: Graph,
    ) -> None:
        self.graph: Graph = graph

    def __call__(self, signal: GraphSig, Lambda=None, Lp=None):
        if signal is None:
            return 0
        term_1 = matrix_dot_product(self.graph.Abar, signal.L)
        # r = self.graph.A - signal.L
        # term_1 = 0.5 * matrix_dot_product(r, r)

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

        # term5
        if Lp is not None:
            d = signal.L - Lp
            term_5 = matrix_dot_product(Lambda, d)
        else:
            term_5 = 0

        # term6
        if Lp is not None and Lambda is not None:
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

    def calc_grad(self, signal, Lambda, Lp) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal, Lambda, Lp)
        grad_R = self.calc_grad_R(signal)
        grad_pi = self.calc_grad_pi(signal)

        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig, Lambda, Lp):
        # return self.graph.A - signal.L
        return self.graph.Abar - Lambda - config.beta * (signal.L - Lp)
