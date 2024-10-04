from atoms_coefs import AtomsCoefs


class Sig:
    def __init__(self, dict: AtomsCoefs) -> None:
        self.dict = dict

    def calc_objective(self):
        return NotImplementedError
