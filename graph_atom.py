import numpy as np
from atom import Atom
from utils import *


class GraphAtom(Atom):
    def __init__(self) -> None:
        super().__init__()

    def set_params(self, e, R, b, coef) -> None:
        self.e = e
        self.R = R
        self.b = b
        self.coef = coef
        self.n = e.shape[0]
        self.d = R.shape[0]
        self.c = b.shape[0]

    def initialize(self, n, d, c):
        self.n = n
        self.d = d
        self.c = c
        # Initialization
        self.e = np.random.randn(self.n)
        self.R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
            self.d,
            config.rho_min,
            config.rho_max,
            noise_level=0.0,
            include_extremes=False,
        )
        self.b = create_multiple_noisy_probability_vectors(1, self.c, noise_level=1)[0]

        aaa = 9

    def e_term(self):
        return np.outer(self.e, self.e)

    def R_term(self):
        return np.einsum("i,jk->ijk", ((self.e**2)), self.R)

    def b_term(self):
        return np.einsum("i,j->ij", ((self.e**2)), self.b)

    def calc_outer_e(self):
        return self.coef * self.e_term()

    def calc_outer_R(self):
        return self.coef * self.R_term()

    def calc_outer_pi(self):
        return self.coef * self.b_term()

    def grad_R(self):
        return np.sum(self.R * (self.e**2)[:, np.newaxis, np.newaxis], axis=0)

    def update_e(self, e):
        self.e = e

    def update_R(self, R):
        self.R = R

    def update_b(self, b):
        self.b = b

    def update_coef(self, coef):
        self.coef = coef
