from sig import Sig


class GraphSig(Sig):
    def __init__(self, L, R, pi) -> None:
        super().__init__(dict)
        self.L = L
        self.Rv = R
        self.pi = pi
