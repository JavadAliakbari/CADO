from Atom.sig import Sig


class Atom:
    def __init__(self) -> None:
        pass
        # self.coef = coef

    def calc_sig_term(self) -> Sig:
        raise NotImplementedError

    def __add__(self, atom):
        raise NotImplementedError

    def __sub__(self, atom):
        raise NotImplementedError

    def __mul__(self, c):
        raise NotImplementedError

    def __rmul__(self, c):
        raise NotImplementedError
