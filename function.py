from atoms_coefs import AtomsCoefs
from data import Data


class f:
    def __init__(self, signal: AtomsCoefs, data: Data) -> None:
        self.signal = signal
        self.graph = data

    def calc_objective(self):
        raise NotImplementedError

    def calc_grad(self):
        raise NotImplementedError
