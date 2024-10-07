from sig import Sig
from utils import *


class GraphSig(Sig):
    def __init__(self, L, R, pi) -> None:
        super().__init__()
        self.L = L
        self.Rv = R
        self.pi = pi
        self.n = L.shape[0]
        self.d = R.shape[-1]
        self.c = self.pi.shape[-1]

    def __add__(self, sig):
        if isinstance(sig, Sig):
            L = self.L + sig.L
            Rv = self.Rv + sig.Rv
            pi = self.pi + sig.pi
        else:
            L = self.L + sig
            Rv = self.Rv + sig
            pi = self.pi + sig
        sig = GraphSig(L, Rv, pi)

        return sig

    def __radd__(self, c):
        L = self.L + c
        Rv = self.Rv + c
        pi = self.pi + c
        sig = GraphSig(L, Rv, pi)

        return sig

    def __sub__(self, sig):
        if isinstance(sig, Sig):
            L = self.L - sig.L
            Rv = self.Rv - sig.Rv
            pi = self.pi - sig.pi
        else:
            L = self.L - sig
            Rv = self.Rv - sig
            pi = self.pi - sig
        sig = GraphSig(L, Rv, pi)
        return sig

    def __mul__(self, c):
        L = c * self.L
        Rv = c * self.Rv
        pi = c * self.pi
        sig = GraphSig(L, Rv, pi)
        return sig

    def __rmul__(self, c):
        L = c * self.L
        Rv = c * self.Rv
        pi = c * self.pi
        sig = GraphSig(L, Rv, pi)
        return sig

    def get_params(self):
        return self.n, self.d, self.c

    def dot(self, sig):
        term1 = matrix_dot_product(self.L, sig.L)
        term2 = matrix_dot_product(self.Rv, sig.Rv)
        term3 = matrix_dot_product(self.pi, sig.pi)

        return -term1 + config.mu1 * term2 - config.mu2 * term3
