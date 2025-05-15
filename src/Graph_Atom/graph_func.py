import numpy as np

from Graph_Atom.graph_sig import GraphSig
from Atom.function import f
from utils.graph import Graph
from utils.utils import data, algorithm, matrix_dot_product


class GraphFunc(f):
    def __init__(self, graph: Graph, **kwargs) -> None:
        self.graph: Graph = graph
        self.mu0 = kwargs.get("mu0", data.mu0)
        self.mu1 = kwargs.get("mu1", data.mu1)
        self.mu2 = kwargs.get("mu2", data.mu2)
        self.p = kwargs.get("p", data.p)
        self.q = kwargs.get("q", data.q)
        self.train_ratio = kwargs.get("train_ratio", data.train_ratio)

        self.grad_pi = self.calc_grad_pi()

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
        normalizer = signal.n if algorithm.normalize else 1
        if algorithm.Abar:
            L_term = -matrix_dot_product(signal.L, self.graph.Abar) / normalizer
        else:
            r = self.graph.A - signal.L
            L_term = 0.5 * matrix_dot_product(r, r) / normalizer

        return self.mu0 * L_term

    def calc_R(self, signal: GraphSig):
        normalizer = signal.d**2 if algorithm.normalize else 1
        if self.mu1 != 0:
            batch_size = 512
            n = signal.Rv.shape[0]
            d = signal.Rv.shape[-1]
            term_2_val = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_Rv = signal.Rv[start:end]
                I_batch = np.broadcast_to(np.eye(d), batch_Rv.shape)
                batch_Rv_inv = np.linalg.solve(batch_Rv, I_batch)
                batch_cov = self.graph.cov[start:end]
                term_2_val += np.einsum("ijk,ijk->", batch_cov, batch_Rv_inv)
            term_2 = term_2_val / normalizer

            # Use slogdet to compute term_3
            _, logdet = np.linalg.slogdet(signal.Rv)
            term_3 = np.sum(logdet) / normalizer
        else:
            term_2 = 0
            term_3 = 0

        return self.mu1 * term_2 + self.mu1 * term_3

    def calc_pi(self, signal: GraphSig):
        normalizer = signal.c if algorithm.normalize else 1
        if self.mu2 != 0:
            train_labels = self.graph.train_labels
            train_nodes = self.graph.train_nodes
            term_4 = np.sum(signal.pi[train_nodes, train_labels]) / normalizer
        else:
            term_4 = 0

        return -self.mu2 * term_4

    def calc_cov(self):
        return self.graph.cov

    def calc_grad(self, signal, need_L=True) -> GraphSig:
        if signal is None:
            return 0
        if need_L:
            grad_L = self.calc_grad_L(signal)
        else:
            grad_L = None
        grad_R = self.calc_grad_R(signal)
        grad_pi = self.grad_pi
        grad = GraphSig(grad_L, grad_R, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig):
        normalizer = signal.n * (self.p) * (1 - self.q) if algorithm.normalize else 1

        if algorithm.Abar:
            return self.mu0 * (-self.graph.Abar) / normalizer
        else:
            return self.mu0 * (-self.graph.A + signal.L) / normalizer

    def calc_grad_R(self, signal: GraphSig):
        normalizer = signal.d**2 if algorithm.normalize else 1
        if self.mu1 != 0:
            d = signal.Rv.shape[-1]
            n = signal.Rv.shape[0]

            batch_size = 512

            gradient_Rv = np.empty_like(signal.Rv)
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_Rv = signal.Rv[start:end]
                I_batch = np.broadcast_to(np.eye(d), batch_Rv.shape)
                X = np.linalg.solve(batch_Rv, I_batch)
                cov_batch = self.graph.cov[start:end]
                Y = np.matmul(cov_batch, X)
                X2 = np.linalg.solve(batch_Rv, Y)
                # gradient_Rv[start:end] = I_batch - X2
                gradient_Rv[start:end] = X - X2
            return self.mu1 * gradient_Rv / normalizer
        else:
            return None

    def calc_grad_pi(self):
        normalizer = (
            self.graph.num_classes * self.train_ratio if algorithm.normalize else 1
        )
        gradient_pi = np.zeros(
            (self.graph.num_nodes, self.graph.num_classes), dtype=float
        )
        train_labels = self.graph.train_labels
        train_nodes = self.graph.train_nodes
        gradient_pi[train_nodes, train_labels] = 1
        gradient_pi /= np.sum(gradient_pi, axis=0)[None, :]

        return -self.mu2 * gradient_pi / normalizer

    def update_mu0(self, new_mu1):
        """Update the mu1 parameter value"""
        self.mu1 = new_mu1

    def calc_grad_mu0(self, signal: GraphSig):
        """Calculate gradient of the function with respect to mu0"""
        normalizer = signal.n if algorithm.normalize else 1
        if algorithm.Abar:
            L_term = -matrix_dot_product(signal.L, self.graph.Abar) / normalizer
        else:
            r = self.graph.A - signal.L
            L_term = 0.5 * matrix_dot_product(r, r) / normalizer

        return L_term  # The gradient is simply the L_term

    def update_mu1(self, new_mu1):
        """Update the mu1 parameter value"""
        self.mu1 = new_mu1

    def calc_grad_mu1(self, signal: GraphSig):
        normalizer = signal.d**2 if algorithm.normalize else 1
        if self.mu1 != 0:
            batch_size = 512
            n = signal.Rv.shape[0]
            d = signal.Rv.shape[-1]
            term_2_val = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_Rv = signal.Rv[start:end]
                I_batch = np.broadcast_to(np.eye(d), batch_Rv.shape)
                batch_Rv_inv = np.linalg.solve(batch_Rv, I_batch)
                batch_cov = self.graph.cov[start:end]
                term_2_val += np.einsum("ijk,ijk->", batch_cov, batch_Rv_inv)
            term_2 = term_2_val / normalizer

            # Use slogdet to compute term_3
            _, logdet = np.linalg.slogdet(signal.Rv)
            term_3 = np.sum(logdet) / normalizer
        else:
            term_2 = 0
            term_3 = 0

        return term_2 + term_3
