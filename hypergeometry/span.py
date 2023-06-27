
import numpy as np

from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.combination import Combination

class Span:
    """This class represents a subspace as a set of points that are spanned by a set of vectors (basis)
    from a given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where -inf < xi < +inf
    """
    
    def __init__(self, org: Point, basis: Poly):
        assert org.dim() == basis.dim()
        self.org = org
        self.basis = basis
    
    def space_dim(self) -> int:
        """The dimensionality of the space we are part of"""
        return self.basis.dim()
    
    def my_dim(self) -> int:
        """The dimensionality of the subspace, assuming the vectors are independent"""
        return self.basis.num()
    
    @classmethod
    def from_combination(cls, comb: Combination) -> 'Span':
        return Span(org=comb.at(0), basis=Poly(comb.v.p[1:] - comb.v.p[0]))
    
    def as_combination(self) -> Combination:
        return Combination(np.concatenate((np.zeros((1, self.space_dim())), self.basis.p), axis=0) + self.org.c)
    
    def allclose(self, o: 'Span'):
        return self.org.allclose(o.org) and self.basis.allclose(o.basis)
    
    
# Unit tests
if __name__ == "__main__":

    p = Point([1,1,1])
    v = Poly([[1,1,0], [0,0,1]])
    span = Span(org=p, basis=v)
    assert span.space_dim() == 3
    assert span.my_dim() == 2
    comb = Combination([[1,1,1], [2,2,1], [1,1,2]])
    assert span.as_combination().allclose(comb)
    assert Span.from_combination(comb).allclose(span)
    assert comb.space_dim() == 3
    assert comb.my_dim() == 2
