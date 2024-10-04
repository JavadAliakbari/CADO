from typing import List
from atoms_coefs import AtomsCoefs
from graph_sig import GraphSig


class GraphAtoms(AtomsCoefs):
    def __init__(self) -> None:
        super().__init__()

    def calc_objective(self) -> GraphSig:
        L = self.calc_L()
        R = self.calc_R()
        pi = self.calc_pi()
        sig = GraphSig(L, R, pi)
        return sig

    def calc_L(self):
        return sum([atom.calc_outer_e() for atom in self.atoms])

    def calc_R(self):
        return sum([atom.calc_outer_R() for atom in self.atoms])

    def calc_pi(self):
        return sum([atom.calc_outer_pi() for atom in self.atoms])
