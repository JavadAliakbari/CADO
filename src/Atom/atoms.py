from typing import List
from Atom.atom import Atom
from Atom.coefs import Coefs
from Atom.sig import Sig
import itertools

from utils.utils import *


class Atoms:
    def __init__(self, atoms=None) -> None:
        self.atoms: List[Atom] = atoms if atoms is not None else []

    def __len__(self):
        return len(self.atoms)

    def __iter__(self):
        return iter(self.atoms)

    def __getitem__(self, idx):
        return self.atoms[idx]

    def __add__(self, atoms_):
        atoms = Atoms()
        for atom1, atom2 in zip(self.atoms, atoms_):
            atom = atom1 + atom2
            atoms.add_atom(atom)

        return atoms

    def __sub__(self, atoms_):
        atoms = Atoms()
        for atom1, atom2 in zip(self.atoms, atoms_):
            atom = atom1 - atom2
            atoms.add_atom(atom)

        return atoms

    def __mul__(self, c):
        atoms = Atoms()
        if isinstance(c, (int, float)):  # Check if c is a scalar
            for atom1 in self.atoms:
                atom = atom1 * c
                atoms.add_atom(atom)
        elif isinstance(
            c, (list, np.ndarray, Coefs)
        ):  # Check if c is a list or np.array
            if len(c) != len(self.atoms):
                raise ValueError("Length of c must match the number of atoms.")
            for atom1, ci in zip(self.atoms, c):
                atom = atom1 * ci
                atoms.add_atom(atom)
        else:
            raise TypeError("c must be a scalar, list, or numpy array.")

        return atoms

    def __rmul__(self, c):
        atoms = Atoms()
        if isinstance(c, (int, float)):  # Check if c is a scalar
            for atom1 in self.atoms:
                atom = atom1 * c
                atoms.add_atom(atom)
        elif isinstance(
            c, (list, np.ndarray, Coefs)
        ):  # Check if c is a list or np.array
            if len(c) != len(self.atoms):
                raise ValueError("Length of c must match the number of atoms.")
            for atom1, ci in zip(self.atoms, c):
                atom = atom1 * ci
                atoms.add_atom(atom)
        else:
            raise TypeError("c must be a scalar, list, or numpy array.")

        return atoms

    # def __rmul__(self, c):
    #     atoms = Atoms()
    #     for atom1 in self.atoms:
    #         atom = atom1 * c
    #         atoms.add_atom(atom)

    #     return atoms

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

        return sig
