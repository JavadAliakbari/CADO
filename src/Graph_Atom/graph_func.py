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
        u_term = self.calc_u(signal)
        pi_term = self.calc_pi(signal)

        obj = L_term + u_term + pi_term

        if components:
            return obj, L_term, u_term, pi_term
        else:
            return obj

    def calc_L(self, signal: GraphSig):
        normalizer = (signal.n**2) if algorithm.normalize else 1
        if algorithm.Abar:
            L_term = -matrix_dot_product(signal.L, self.graph.Abar) / normalizer
        else:
            r = self.graph.A - signal.L
            L_term = 0.5 * matrix_dot_product(r, r) / normalizer

        return self.mu0 * L_term

    def calc_u(self, signal: GraphSig):
        """K-means objective: 0.5 * sum_v ||x_v - u_v||^2"""
        normalizer = signal.d if algorithm.normalize else 1
        if self.mu1 != 0 and signal.u_v is not None:
            # Compute the squared distance between features and cluster centers
            # X: (n x d), u_v: (n x d)
            diff = self.graph.X - signal.u_v  # (n x d)
            kmeans_term = 0.5 * np.mean(diff**2) / normalizer
        else:
            kmeans_term = 0

        return self.mu1 * kmeans_term

    def calc_pi(self, signal: GraphSig):
        normalizer = signal.c if algorithm.normalize else 1
        if self.mu2 != 0:
            train_labels = self.graph.train_labels
            train_nodes = self.graph.train_nodes
            term_4 = np.sum(signal.pi[train_nodes, train_labels]) / normalizer
            return -self.mu2 * term_4 / train_nodes.shape[0]
        else:
            return 0

    def calc_grad(self, signal, need_L=True) -> GraphSig:
        if signal is None:
            return 0
        if need_L:
            grad_L = self.calc_grad_L(signal)
        else:
            grad_L = None
        grad_u = self.calc_grad_u(signal)
        grad_pi = self.grad_pi
        grad = GraphSig(grad_L, grad_u, grad_pi)
        return grad

    def calc_grad_L(self, signal: GraphSig):
        normalizer = (
            (signal.n**2) * (self.p) * (1 - self.q) if algorithm.normalize else 1
        )

        if algorithm.Abar:
            return self.mu0 * (-self.graph.Abar) / normalizer
        else:
            return self.mu0 * (-self.graph.A + signal.L) / normalizer

    def calc_grad_u(self, signal: GraphSig):
        """Gradient of k-means objective w.r.t. u_v: -(x_v - u_v)"""
        normalizer = signal.d if algorithm.normalize else 1
        if self.mu1 != 0 and signal.u_v is not None:
            # Gradient w.r.t. u_v is: -(X - u_v)
            diff = self.graph.X - signal.u_v  # (n x d)
            gradient_u_v = -diff / signal.n  # (n x d)
            return self.mu1 * gradient_u_v / normalizer
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
        gradient_pi /= np.sum(gradient_pi, axis=0)[None, :] + 1e-10

        return -self.mu2 * gradient_pi / normalizer

    def update_mu0(self, new_mu0):
        """Update the mu0 parameter value"""
        self.mu0 = new_mu0

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
        """Gradient w.r.t. mu1 for k-means objective"""
        normalizer = signal.d if algorithm.normalize else 1
        if self.mu1 != 0 and signal.u_v is not None:
            # Compute the k-means term
            diff = self.graph.X - signal.u_v
            kmeans_term = 0.5 * np.mean(diff**2) / normalizer
        else:
            kmeans_term = 0

        return kmeans_term
