from sig import Sig


class Atom:
    def __init__(self) -> None:
        pass
        # self.coef = coef

    def calc_sig_term(self) -> Sig:
        raise NotImplementedError

    # def set_params(self, coef) -> None:
    #     self.coef = coef
