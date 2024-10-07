from typing import List
from atom import Atom
from coefs import Coefs
from sig import Sig


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

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def remove_atom(self, atom: Atom):
        self.atoms.remove(atom)

    def calc_sig(self, coefs: Coefs) -> Sig:
        return sum(
            [c * atom.calc_sig_term() for atom, c in zip(self.atoms, coefs.get_coefs())]
        )
