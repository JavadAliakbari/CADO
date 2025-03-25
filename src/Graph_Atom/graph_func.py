from Graph_Atom.graph_sig import GraphSig
from Atom.function import f
from utils.graph import Graph
from utils.utils import *


class GraphFunc(f):
    def __init__(self, graph: Graph, mu1=config.mu1, mu2=config.mu2) -> None:
        self.graph: Graph = graph
        self.admm = None
        self.mu1 = mu1
        self.mu2 = mu2

    def set_ADMM_params(self, admm):
        self.admm = admm

    def __call__(self, signal: GraphSig, ADMM=config.ADMM, components=False):
        if signal is None:
            return 0

        L_term = self.calc_L(signal)
        R_term = self.calc_R(signal)
        pi_term = self.calc_pi(signal)

        # term5
        if self.admm is not None and ADMM == "L":
            admm_term = self.admm.admm_value(signal.L)
        elif self.admm is not None and ADMM == "pi":
            admm_term = self.admm.admm_value(signal.pi)
        else:
            admm_term = 0

        obj = L_term + R_term + pi_term + admm_term

        if components:
            return obj, L_term, R_term, pi_term, admm_term
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

    def calc_grad(self, signal, ADMM=config.ADMM) -> GraphSig:
        if signal is None:
            return 0
        grad_L = self.calc_grad_L(signal, ADMM=ADMM)
        normalizer = signal.c * config.train_ratio if config.normalize else 1
        grad_R = self.calc_grad_R(signal) / normalizer
        grad_pi = self.calc_grad_pi(signal, ADMM=ADMM) / normalizer
        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig, ADMM=config.ADMM):
        normalizer = signal.n * config.p if config.normalize else 1
        if self.admm is not None and ADMM == "L":
            grad_admm = self.admm.calc_grad(signal.L)
        else:
            grad_admm = 0

        if config.Abar:
            return (-self.graph.Abar) / normalizer + grad_admm
        else:
            return (-self.graph.A + signal.L) / normalizer + grad_admm
        # return self.graph.A

    # def calc_grad_R(self, signal: GraphSig):
    #     if signal.Rz is None:
    #         signal.solve_R(self.graph.X)

    #     results = np.einsum("ij,ik->ijk", signal.Rz, signal.Rz)
    #     gradient_Rv = np.eye(self.graph.X.shape[1])[None, :] - results

    #     return self.mu1 * gradient_Rv

    def calc_grad_R(self, signal: GraphSig):
        # Second block: Gradient with respect to Rv_matrices
        if signal.Rv_inv is None:
            signal.calc_Rv_inv()

        # gradient_Rv = Rv_inv - np.einsum(
        #     "ijk,ikm,imn->ijn", Rv_inv, self.graph.cov, Rv_inv
        # )
        # gradient_Rv = np.eye(self.graph.num_features)[None, :] - np.einsum(
        #     "ijk,ikm,imn->ijn", signal.Rv_inv, self.graph.cov, signal.Rv_inv
        # )
        gradient_Rv = signal.Rv_inv - np.einsum(
            "ijk,ikm,imn->ijn", signal.Rv_inv, self.graph.cov, signal.Rv_inv
        )

        return self.mu1 * gradient_Rv

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

        return -self.mu2 * gradient_pi - grad_admm
