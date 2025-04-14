from Atom.sig import Sig
from utils.utils import *


class GraphSig(Sig):
    def __init__(self, L, Rv, pi) -> None:
        super().__init__()
        self.c = pi.shape[-1]
        self.d = 0

        if L is not None:
            self.L = L
            self.n = L.shape[0]
        else:
            self.L = None
            self.n = 0

        if Rv is not None:
            self.d = Rv.shape[-1]
            eyes = np.eye(self.d)[None, :]
            self.Rv = Rv
        else:
            self.Rv = None
            self.d = None
        self.pi = pi

        self.Rv_inv = None
        self.Rz = None

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
        elif isinstance(sig, (int, float)):
            L = self.L - sig
            Rv = self.Rv - sig
            pi = self.pi - sig
        sig = GraphSig(L, Rv, pi)
        return sig

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            L = c * self.L
            Rv = c * self.Rv
            pi = c * self.pi
            sig = GraphSig(L, Rv, pi)
            return sig
        elif isinstance(c, GraphSig):
            return self.dot(c)

    def __rmul__(self, c):
        L = c * self.L
        Rv = c * self.Rv
        pi = c * self.pi
        sig = GraphSig(L, Rv, pi)
        return sig

    def __truediv__(self, c):
        if c == 0:
            raise ValueError("Division by zero is not allowed")
        L = self.L / c
        if self.Rv is not None:
            Rv = self.Rv / c
        pi = self.pi / c
        sig = GraphSig(L, Rv, pi)
        return sig

    def __rtruediv__(self, c):
        if isinstance(c, (int, float)):
            L = c / self.L
            Rv = c / self.Rv if self.Rv is not None else None
            pi = c / self.pi
            sig = GraphSig(L, Rv, pi)
            return sig
        else:
            raise ValueError("Right division is only supported for scalars")

    def get_params(self):
        return self.n, self.d, self.c

    def dot(self, sig):
        term1 = matrix_dot_product(self.L, sig.L)
        if self.Rv is not None and sig.Rv is not None:
            term2 = matrix_dot_product(self.Rv, sig.Rv)
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


class EmptySig(GraphSig):
    def __init__(self, n, d, c) -> None:
        self.n = n
        self.d = d
        self.c = c

        self.L = np.zeros((self.n, self.n), dtype=float)
        self.Rv = np.array([data.rho_min * np.eye(self.d) for _ in range(self.n)])
        self.pi = np.array([np.zeros(c) for _ in range(self.n)])
        self.Rv_inv = None
        self.Rz = None
