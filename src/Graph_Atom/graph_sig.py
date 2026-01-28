from Atom.sig import Sig
from utils.utils import *


class GraphSig(Sig):
    def __init__(self, L, u_v, pi) -> None:
        super().__init__()
        self.c = pi.shape[-1]
        self.d = 0

        if L is not None:
            self.L = L
            self.n = L.shape[0]
        else:
            self.L = None
            self.n = 0

        if u_v is not None:
            self.d = u_v.shape[-1]
            self.u_v = u_v
        else:
            self.u_v = None
            self.d = None
        self.pi = pi

    def __add__(self, sig):
        if isinstance(sig, Sig):
            L = self.L + sig.L
            u_v = self.u_v + sig.u_v
            pi = self.pi + sig.pi
        else:
            L = self.L + sig
            u_v = self.u_v + sig
            pi = self.pi + sig
        sig = GraphSig(L, u_v, pi)

        return sig

    def __radd__(self, c):
        L = self.L + c
        u_v = self.u_v + c
        pi = self.pi + c
        sig = GraphSig(L, u_v, pi)

        return sig

    def __sub__(self, sig):
        if isinstance(sig, Sig):
            L = self.L - sig.L
            u_v = self.u_v - sig.u_v
            pi = self.pi - sig.pi
        elif isinstance(sig, (int, float)):
            L = self.L - sig
            u_v = self.u_v - sig
            pi = self.pi - sig
        sig = GraphSig(L, u_v, pi)
        return sig

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            L = c * self.L
            u_v = c * self.u_v
            pi = c * self.pi
            sig = GraphSig(L, u_v, pi)
            return sig
        elif isinstance(c, GraphSig):
            return self.dot(c)

    def __rmul__(self, c):
        L = c * self.L
        u_v = c * self.u_v
        pi = c * self.pi
        sig = GraphSig(L, u_v, pi)
        return sig

    def __truediv__(self, c):
        if c == 0:
            raise ValueError("Division by zero is not allowed")
        L = self.L / c
        if self.u_v is not None:
            u_v = self.u_v / c
        pi = self.pi / c
        sig = GraphSig(L, u_v, pi)
        return sig

    def __rtruediv__(self, c):
        if isinstance(c, (int, float)):
            L = c / self.L
            u_v = c / self.u_v if self.u_v is not None else None
            pi = c / self.pi
            sig = GraphSig(L, u_v, pi)
            return sig
        else:
            raise ValueError("Right division is only supported for scalars")

    def __matmul__(self, sig):
        return self.dot(sig)

    def get_params(self):
        return self.n, self.d, self.c

    def dot(self, sig):
        term1 = matrix_dot_product(self.L, sig.L)
        if self.u_v is not None and sig.u_v is not None:
            term2 = matrix_dot_product(self.u_v, sig.u_v)
        else:
            term2 = 0
        term3 = matrix_dot_product(self.pi, sig.pi)
        return term1 + term2 + term3

    def update(self, grad, lr):
        self.L -= lr * grad.L
        if self.Rv is not None and grad.Rv is not None:
            self.Rv -= lr * grad.Rv
        self.pi -= lr * grad.pi

    def project(self):
        self.L = project_to_bound(self.L)
        # if self.Rv is not None:
        #     self.Rv = project(self.Rv)
        # self.pi = project(self.pi)
