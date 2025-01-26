from typing import List
from Atom.atom import Atom
from Atom.coefs import Coefs
from Atom.sig import Sig
import itertools

from utils.utils import *


class Atoms:
    def __init__(
        self,
    ) -> None:
        self.atoms: List[Atom] = []

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        return iter(self.atoms)

    def __getitem__(self, idx):
        return self.atoms[idx]

    def shuffle(self):
        np.random.shuffle(self.atoms)

    def rearnge(self, idxs):
        self.atoms = [self.atoms[i] for i in idxs]

    def __setitem__(self, idx, atom):
        self.atoms[idx] = atom

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def remove_atom(self, atom: Atom):
        self.atoms.remove(atom)

    def remove_idx(self, idx):
        del self.atoms[idx]

    def calc_sig(self, coefs: Coefs) -> Sig:
        if len(coefs) == 0:
            return None
        sig: Sig = sum(
            [c * atom.calc_sig_term() for atom, c in zip(self.atoms, coefs.get_coefs())]
        )

        sig.Rv += config.rho_min * np.eye(sig.Rv.shape[-1])[None, :]

        return sig
