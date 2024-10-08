from copy import deepcopy
import numpy as np


class Coefs:
    def __init__(self, coefs=None) -> None:
        if coefs is not None:
            self.coefs = coefs
        else:
            self.coefs = []

    def __len__(self):
        return len(self.coefs)

    def __iter__(self):
        return iter(self.coefs)

    def __getitem__(self, idx):
        return self.coefs[idx]

    def scale_coefficients(self, factor):
        self.coefs[:-1] *= 1 - factor
        self.coefs[-1] *= factor

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
