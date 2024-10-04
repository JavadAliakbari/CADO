class Atom:
    def __init__(self, coef=0) -> None:
        self.coef = coef

    def set_params(self, e, R, b, coef) -> None:
        self.coef = coef
