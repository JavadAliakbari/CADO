from Atom.sig import Sig
from utils.utils import *


class GraphSig(Sig):
    def __init__(self, L, Rv, pi, add_eps=True) -> None:
        super().__init__()
        self.n = L.shape[0]
        self.c = pi.shape[-1]
        self.d = Rv.shape[-1]

        self.L = L
        if Rv is not None:
            self.d = Rv.shape[-1]
            eyes = np.eye(self.d)[None, :]
            self.Rv = Rv
            if add_eps:
                self.Rv += config.eps * eyes
        else:
            self.Rv = None
            self.d = None
        self.pi = pi

        self.Rv_inv = None
        self.Rz = None

    def calc_Rv_inv(self):
        self.Rv_inv = np.array([invert_matrix(matrix) for matrix in self.Rv])

    def solve_R(self, X):
        # To store the final results
        self.Rz = np.empty_like(X)
        for i, (R, x) in enumerate(zip(self.Rv, X)):
            # Solve R @ y = x for y, which gives y = R^{-1} @ x
            try:
                z = np.linalg.solve(R, x)
                # z = np.linalg.lstsq(R, x, rcond=None)[0]
            except:
                z = x

            self.Rz[i] = z

    def __add__(self, sig):
        if isinstance(sig, Sig):
            L = self.L + sig.L
            Rv = self.Rv + sig.Rv
            pi = self.pi + sig.pi
        else:
            L = self.L + sig
            Rv = self.Rv + sig
            pi = self.pi + sig
        sig = GraphSig(L, Rv, pi, add_eps=False)

        return sig

    def __radd__(self, c):
        L = self.L + c
        Rv = self.Rv + c
        pi = self.pi + c
        sig = GraphSig(L, Rv, pi, add_eps=False)

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
        sig = GraphSig(L, Rv, pi, add_eps=False)
        return sig

    def __mul__(self, c):
        L = c * self.L
        Rv = c * self.Rv
        pi = c * self.pi
        sig = GraphSig(L, Rv, pi, add_eps=False)
        return sig

    def __rmul__(self, c):
        L = c * self.L
        Rv = c * self.Rv
        pi = c * self.pi
        sig = GraphSig(L, Rv, pi, add_eps=False)
        return sig

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


class EmptySig(GraphSig):
    def __init__(self, n, d, c) -> None:
        self.n = n
        self.d = d
        self.c = c

        self.L = np.zeros((self.n, self.n), dtype=float)
        self.Rv = np.array([config.rho_min * np.eye(self.d) for _ in range(self.n)])
        self.pi = np.array([np.zeros(c) / c for _ in range(self.n)])
        self.Rv_inv = None
        self.Rz = None
