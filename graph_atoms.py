from typing import List

import numpy as np
from atoms_coefs import AtomsCoefs
from graph_atom import GraphAtom
from graph_sig import GraphSig


class GraphAtoms(AtomsCoefs):
    def __init__(self) -> None:
        super().__init__()

    def calc_sig(self) -> GraphSig:
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

    def scale_coefficients(self, factor):
        atom: GraphAtom
        for atom in self.atoms[:-1]:
            atom.update_coef(atom.coef * (1 - factor))

        last_atom: GraphAtom = self.atoms[-1]
        last_atom.update_coef(last_atom.coef * factor)

    def update_coefficients(self, coefs=None):
        atom: GraphAtom
        for atom, c in zip(self.atoms, coefs):
            atom.update_coef(c)

    def get_coefs(self):
        return np.array([atom.coef for atom in self.atoms])
