import numpy as np

from utils.SBM import SparseAbar


# Add a lazy outer product helper for rank-1 matrices
class OuterProduct:
    def __init__(self, vector, coef=1):
        self.vector = vector
        self.coef = coef

    @property
    def shape(self):
        n = self.vector.shape[0]
        return (n, n)

    def __array__(self, dtype=None):
        return self.coef * np.outer(self.vector, self.vector)

    # Added: efficient lazy addition functionality
    def __add__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum([self, other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum([self] + other.terms)
        elif isinstance(other, (int, float)):
            if other == 0:
                return self
            else:
                return self.__array__() + other
        else:
            return self.__array__() + other

    def __sub__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum([self, -1 * other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum([self] + (-1 * other).terms)
        elif isinstance(other, (int, float)):
            if other == 0:
                return self
            else:
                return self.__array__() + other
        else:
            return self.__array__() + other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return OuterProduct(self.vector, self.coef * c)
        elif isinstance(c, np.ndarray) or c.__class__.__name__ == "SparseAbar":
            # Faster implementation using two dot products instead of np.einsum
            dot_product = c @ self.vector
            return self.coef * (self.vector @ dot_product)

    def __rmul__(self, c):
        return self * c


# New class to lazily sum multiple OuterProduct objects
class LazyOuterSum:
    def __init__(self, terms):
        self.terms = terms  # list of OuterProduct instances

    @property
    def shape(self):
        n = self.terms[0].vector.shape[0]
        return (n, n)

    def __array__(self, dtype=None):
        total = 0
        for term in self.terms:
            total += np.array(term)
        return total

    def __add__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum(self.terms + [other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum(self.terms + other.terms)
        else:
            return np.array(self) + other

    def __sub__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum(self.terms + [-1 * other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum(self.terms + [-1 * term for term in other.terms])
        else:
            return np.array(self) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return LazyOuterSum([term * c for term in self.terms])
        elif isinstance(c, np.ndarray) or c.__class__.__name__ == "SparseAbar":
            return sum([term * c for term in self.terms])
        else:
            return np.array(self) * c

    def __rmul__(self, c):
        return self * c
