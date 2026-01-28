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

    def __truediv__(self, c):
        raise NotImplementedError

    def dot(self, sig):
        raise NotImplementedError

    def project(self):
        raise NotImplementedError
