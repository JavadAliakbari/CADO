from sig import Sig


class GraphSig(Sig):
    def __init__(self, L, R, pi) -> None:
        super().__init__()
        self.L = L
        self.Rv = R
        self.pi = pi

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
