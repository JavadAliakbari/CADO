from typing import List
from atom import Atom


class AtomsCoefs:
    def __init__(
        self,
    ) -> None:
        self.atoms: List[Atom] = []

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def remove_atom(self, atom: Atom):
        self.atoms.remove(atom)

    def calc_objective(self):
        sum([])


# type: ignore
