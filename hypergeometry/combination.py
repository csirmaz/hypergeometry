
import numpy as np

from hypergeometry.poly import Poly

class Combination:
    """This class represents a subspace as the collection of points that are the linear combination
    of the vectors:
    X = x0*V0 + ... + xn*Vn
    where x0 + ... + xn = 1
    """
    
    def __init__(self, vectors):
        if not isinstance(vectors, Poly):
            vectors = Poly(vectors)
        self.v = vectors

    def space_dim(self) -> int:
        """The dimensionality of the space we are part of"""
        return self.v.dim()
    
    def my_dim(self) -> int:
        """The dimensionality of the subspace, assuming the vectors are independent"""
        return self.v.num() - 1

    def at(self, x: int) -> 'Point':
        """Return the x'th vector as a Point object"""
        return self.v.at(x)
    
    def allclose(self, o: 'Combination'):
        return self.v.allclose(o.v)

