from copy import deepcopy
import numpy as np

from utils.utils import *


class Coefs:
    def __init__(self, coefs=None) -> None:
        if coefs is not None:
            self.update_coefficients(coefs)
        else:
            self.coefs = []

    def __len__(self):
        return len(self.coefs)

    def __iter__(self):
        return iter(self.coefs)

    def __getitem__(self, idx):
        return self.coefs[idx]

    def __setitem__(self, idx, value):
        self.coefs[idx] = value

    def __lt__(self, other):
        if isinstance(other, Coefs):
            return self.coefs < other.coefs
        return self.coefs < other

    @property
    def ndim(self):
        return self.coefs.ndim  # Return the number of dimensions

    def scale_coefficients(self, factor):
        # s = np.sum(self.coefs)
        self.coefs[:-1] *= 1 - factor
        self.coefs[-1] *= factor

        self.coefs = project_to_simplex2(self.coefs, config.tau)

    def update_coefficients(self, coefs):
        if not isinstance(coefs, np.ndarray):
            self.coefs = np.array(coefs, dtype=np.float32)
        else:
            self.coefs = coefs

    def get_coefs(self):
        return self.coefs

    def add_coef(self, c):
        self.coefs = np.append(self.coefs, c)

    def remove_coef(self, idx):
        # del self.coefs[idx]
        self.coefs = np.delete(self.coefs, idx)

    def scale_coefs(coefs, factor):
        coefs = deepcopy(coefs)
        coefs.scale_coefficients(factor)
        return coefs
