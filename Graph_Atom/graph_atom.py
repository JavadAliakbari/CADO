import numpy as np
from Atom.atom import Atom
from Graph_Atom.graph_sig import GraphSig
from utils.utils import *


class GraphAtom(Atom):
    def __init__(self) -> None:
        super().__init__()

    def set_params(self, e, R, b) -> None:
        self.e = e
        self.R = R
        self.b = b
        # self.coef = coef
        self.n = e.shape[0]
        self.d = R.shape[0]
        self.c = b.shape[0]

    def initialize(self, n, d, c):
        self.n = n
        self.d = d
        self.c = c
        # Initialization
        # self.e = np.zeros(self.n)
        # self.e[:4] = 0.5
        self.e = np.random.randn(self.n)
        # self.e = 0.5 + 0.5 * np.random.randn(self.n)
        # self.e = np.ones(self.n)
        self.e /= np.linalg.norm(self.e)
        self.R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
            self.d,
            0,
            config.rho_max - config.rho_min,
            noise_level=0.0,
        )
        self.b = create_multiple_noisy_probability_vectors(1, self.c, noise_level=1)[0]

        self.R = np.zeros((self.d, self.d))
        self.b = np.zeros(self.c)

    def e_term(self):
        return np.outer(self.e, self.e)

    def R_term(self):
        if self.R is not None:
            return np.einsum("i,jk->ijk", ((self.e**2)), self.R)
        else:
            return None

    def b_term(self):
        return np.einsum("i,j->ij", ((self.e**2)), self.b)

    def calc_sig_term(self) -> GraphSig:
        e_term = self.e_term()
        R_term = self.R_term()
        b_term = self.b_term()
        sig = GraphSig(e_term, R_term, b_term, add_eps=True)
        return sig

    def update_e(self, e):
        self.e = e

    def update_R(self, R):
        self.R = R

    def update_b(self, b):
        self.b = b
