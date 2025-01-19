from Graph_Atom.graph_sig import GraphSig
from Atom.function import f
from utils.graph import Graph
from utils.utils import *


class GraphFunc(f):
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

    # def __call__(self, signal: GraphSig):
    #     if signal is None:
    #         return 0
    #     # term_1 = -matrix_dot_product(self.graph.A, signal.L)
    #     r = self.graph.A - signal.L
    #     term_1 = 0.5 * matrix_dot_product(r, r)

    #     # term2
    #     if signal.Rz is None:
    #         signal.solve_R(self.graph.X)

    #     term_2 = matrix_dot_product(self.graph.X, signal.Rz)

    #     # term3
    #     # term_3 = np.sum(np.log(np.linalg.det(signal.Rv)))
    #     term_3 = np.trace(np.sum(signal.Rv, axis=0))

    #     # term4
    #     train_labels = self.graph.y[self.graph.train_mask]
    #     train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
    #     term_4 = np.sum(signal.pi[train_nodes, train_labels])
    #     obj = term_1 + config.mu1 * term_2 + config.mu1 * term_3 - config.mu2 * term_4

    #     return obj

    def __call__(self, signal: GraphSig, ADMM=config.ADMM):
        if signal is None:
            return 0

        if config.Abar:
            term_1 = -matrix_dot_product(self.graph.Abar, signal.L) / signal.n
        else:
            r = self.graph.A - signal.L
            term_1 = 0.5 * matrix_dot_product(r, r) / signal.n

        # term2
        if config.mu1 != 0:
            if signal.Rv_inv is None:
                signal.calc_Rv_inv()
            # tt = np.einsum("ij,ijk->ik", self.graph.X, signal.Rv_inv)
            # term_2 = matrix_dot_product(tt, self.graph.X)
            # term_2 = np.trace(self.graph.cov @ signal.Rv_inv)
            term_2 = np.einsum("ijk,ijk->", self.graph.cov, signal.Rv_inv) / signal.d

            # term3
            # term_3 = np.sum(np.log(np.linalg.det(signal.Rv)))
            term_3 = np.trace(np.sum(signal.Rv, axis=0)) / signal.d
        else:
            term_2 = 0
            term_3 = 0

        # term4
        if config.mu2 != 0:
            train_labels = self.graph.y[self.graph.train_mask]
            train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
            term_4 = np.sum(signal.pi[train_nodes, train_labels]) / signal.c
        else:
            term_4 = 0

        # term5
        if ADMM and self.Lp is not None:
            d = signal.L - self.Lp
            term_5 = matrix_dot_product(self.Lambda, d) / signal.n
        else:
            term_5 = 0

        # term6
        if ADMM and self.Lp is not None and self.Lambda is not None:
            term_6 = matrix_dot_product(d, d) / signal.n
        else:
            term_6 = 0

        obj = (
            term_1
            + config.mu1 * term_2
            + config.mu1 * term_3
            - config.mu2 * term_4
            + term_5
            + (config.beta / 2) * term_6
        )

        return obj

    def calc_cov(self):
        return self.graph.cov

    def calc_grad(self, signal, ADMM=config.ADMM) -> GraphSig:
        if signal is None:
            return 0
        # grad_L = self.calc_grad_L(signal, ADMM=ADMM)
        # grad_R = self.calc_grad_R(signal)
        # grad_pi = self.calc_grad_pi(signal)
        grad_L = self.calc_grad_L(signal, ADMM=ADMM) / signal.n
        grad_R = self.calc_grad_R(signal) / signal.d
        grad_pi = self.calc_grad_pi(signal) / signal.c

        grad = GraphSig(grad_L, grad_R, grad_pi, add_eps=False)
        return grad

    def calc_grad_L(self, signal: GraphSig, ADMM=config.ADMM):

        if ADMM and self.Lambda is not None:
            grad_lambda = self.Lambda
        else:
            grad_lambda = 0

        if ADMM and self.Lp is not None:
            grad_Lp = self.Lp
        else:
            grad_Lp = 0

        if config.Abar:
            return -self.graph.Abar + grad_lambda + config.beta * (signal.L - grad_Lp)
        else:
            return (
                -self.graph.A
                + signal.L
                + grad_lambda
                + config.beta * (signal.L - grad_Lp)
            )
        # return self.graph.A

    # def calc_grad_R(self, signal: GraphSig):
    #     if signal.Rz is None:
    #         signal.solve_R(self.graph.X)

    #     results = np.einsum("ij,ik->ijk", signal.Rz, signal.Rz)
    #     gradient_Rv = np.eye(self.graph.X.shape[1])[None, :] - results

    #     return config.mu1 * gradient_Rv

    def calc_grad_R(self, signal: GraphSig):
        # Second block: Gradient with respect to Rv_matrices
        if signal.Rv_inv is None:
            signal.calc_Rv_inv()

        # gradient_Rv = Rv_inv - np.einsum(
        #     "ijk,ikm,imn->ijn", Rv_inv, self.graph.cov, Rv_inv
        # )
        gradient_Rv = np.eye(self.graph.num_features)[None, :] - np.einsum(
            "ijk,ikm,imn->ijn", signal.Rv_inv, self.graph.cov, signal.Rv_inv
        )

        return config.mu1 * gradient_Rv

    def calc_grad_pi(self, signal):
        # Third block: Gradient with respect to pi_kv
        gradient_pi = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        # Create a one-hot encoded matrix
        train_labels = self.graph.y[self.graph.train_mask]
        train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
        gradient_pi[train_nodes, train_labels] = 1

        return -config.mu2 * gradient_pi
