
from typing import Union, Iterable
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
    
    def apply_to(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Return the absolute coordinates of the point(s) represented relative to this span"""
        return self.basis.apply_to(subject).add(self.org)

    def extract_from(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Represent the point(s) in `subject` relative to this span"""
        return self.basis.extract_from(subject.sub(self.org))
    
    def rotate(self, coords: Iterable[int], rad: float, around_origin: bool = False) -> 'Span':
        new_org = self.org
        if around_origin:
            new_org = new_org.rotate(coords=coords, rad=rad)
        return Span(org=new_org, basis=self.basis.rotate(coords=coords, rad=rad))
    
