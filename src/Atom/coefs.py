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

    def __add__(self, coefs_):
        coefs = []
        for c1, c2 in zip(self.coefs, coefs_):
            c = c1 + c2
            coefs.append(c)

        return Coefs(coefs)

    def __radd__(self, c_):
        coefs = []
        for c1 in self.coefs:
            c = c1 + c_
            coefs.append(c)

        return Coefs(coefs)

    def __sub__(self, coefs_):
        if isinstance(coefs_, (int, float)):  # Check if c is a scalar
            coefs = []
            for c1 in self.coefs:
                c = c1 - coefs_
                coefs.append(c)

            return Coefs(coefs)
        elif isinstance(
            coefs_, (list, np.ndarray, Coefs)
        ):  # Check if c is a list or np.array
            coefs = []
            for c1, c2 in zip(self.coefs, coefs_):
                c = c1 - c2
                coefs.append(c)

        return Coefs(coefs)

    def __rsub__(self, coefs_):
        coefs = []
        for c1 in self.coefs:
            c = coefs_ - c1
            coefs.append(c)

        return Coefs(coefs)

    def __mul__(self, c_):
        if isinstance(c_, (float, int)):
            coefs = []
            for c1 in self.coefs:
                c = c1 * c_
                coefs.append(c)

            return Coefs(coefs)
        else:
            return c_ * self

    def __rmul__(self, c_):
        coefs = []
        for c1 in self.coefs:
            c = c1 * c_
            coefs.append(c)

        return Coefs(coefs)

    def __truediv__(self, c_):
        if isinstance(c_, (int, float)):  # Check if c is a scalar
            if c_ == 0:
                raise ValueError("Division by zero is not allowed")
            coefs = []
            for c1 in self.coefs:
                c = c1 / c_
                coefs.append(c)
        elif isinstance(
            c_, (list, np.ndarray, Coefs)
        ):  # Check if c is a list or np.array
            if len(c_) != len(self.coefs):
                raise ValueError("Length of c must match the number of atoms.")
            coefs = []
            for c1, ci in zip(self.coefs, c_):
                c = c1 / (ci + 1e-10)
                coefs.append(c)
        else:
            raise TypeError("c must be a scalar, list, or numpy array.")

        return Coefs(coefs)

    def shuffle(self):
        np.random.shuffle(self.coefs)

    def rearnge(self, idxs):
        self.coefs = [self.coefs[i] for i in idxs]

    def __lt__(self, other):
        if isinstance(other, Coefs):
            return self.coefs < other.coefs
        return self.coefs < other

    @property
    def ndim(self):
        return 1

    def scale_coefficients(self, factor, id):
        # s = np.sum(self.coefs)
        n = np.sum(self.coefs)
        for ind in range(len(self.coefs)):
            if ind == id:
                self.coefs[ind] *= factor
            else:
                self.coefs[ind] *= 1 - factor


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
