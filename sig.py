class Sig:
    def __init__(self) -> None:
        pass

    def __add__(self, sig):
        raise NotImplementedError

    def __sub__(self, sig):
        raise NotImplementedError

    def __mul__(self, c):
        raise NotImplementedError

    def __rmul__(self, c):
        raise NotImplementedError
