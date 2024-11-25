from atoms import Atoms
from data import Data
from sig import Sig


class f:
    def __init__(self, signal: Atoms, data: Data) -> None:
        self.signal = signal
        self.graph = data

    def __call__(self):
        raise NotImplementedError

    def calc_grad(self, sig: Sig):
        raise NotImplementedError
