from Atom.coefs import Coefs
from Atom.function import f
from Graph_Atom.graph_sig import GraphSig
from utils.graph import Graph
from utils.utils import *


class GraphFunc_ADOP(f):
    def __init__(
        self,
        graph: Graph,
        coefs: Coefs = None,
    ) -> None:
        self.graph: Graph = graph
        self.admm = None
        self.coefs = coefs

    def set_ADMM_params(self, admm):
        self.admm = admm

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
        if self.admm is not None and ADMM == "L":
            admm_term = self.admm.admm_value(signal.L)
        elif self.admm is not None and ADMM == "pi":
            admm_term = self.admm.admm_value(signal.pi)
        else:
            admm_term = 0

        if config.mu0 != 0:
            term_5 = np.sum(self.coefs.get_coefs()) / signal.c

        obj = (
            term_1
            + config.mu1 * term_2
            + config.mu1 * term_3
            - config.mu2 * term_4
            + admm_term
            + config.mu0 * term_5
        )

        return obj

    def calc_cov(self):
        return self.graph.cov

    def calc_grad(self, signal, ADMM=config.ADMM) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal, ADMM=ADMM) / signal.n
        grad_R = self.calc_grad_R(signal) / signal.d
        grad_pi = self.calc_grad_pi(signal, ADMM=ADMM) / signal.c

        grad = GraphSig(grad_L, grad_R, grad_pi, add_eps=False)
        return grad



    def calc_grad_ADOP(self, signal, ADMM=config.ADMM) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal, ADMM=ADMM) / signal.n
        grad_R = self.calc_grad_R(signal) / signal.d
        grad_pi = self.calc_grad_pi(signal, ADMM=ADMM) / signal.c

        grad = GraphSig(grad_L, grad_R, grad_pi, add_eps=False)
        return grad

    def calc_grad_L(self, signal: GraphSig, ADMM=config.ADMM):
        if self.admm is not None and ADMM == "L":
            grad_admm = self.admm.calc_grad(signal.L)
        else:
            grad_admm = 0

        if config.Abar:
            return -self.graph.Abar + grad_admm
        else:
            return -self.graph.A + signal.L + grad_admm
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

    def calc_grad_pi(self, signal, ADMM=config.ADMM):
        # Third block: Gradient with respect to pi_kv
        gradient_pi = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )  # Assuming labels are zero-indexed and y.max() gives the highest class index
        # Create a one-hot encoded matrix
        train_labels = self.graph.y[self.graph.train_mask]
        train_nodes = np.arange(self.graph.num_nodes)[self.graph.train_mask]
        gradient_pi[train_nodes, train_labels] = 1

        if self.admm is not None and ADMM == "pi":
            grad_admm = self.admm.calc_grad(signal.pi)
        else:
            grad_admm = 0

        return -config.mu2 * gradient_pi - grad_admm
