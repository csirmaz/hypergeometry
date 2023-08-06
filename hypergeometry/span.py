
from typing import Union, Iterable, Any
Self=Any
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
        
    @classmethod
    def create_line(cls, org: Any, direction: Any) -> Self:
        """Convenience function to create a Span representing a line, e.g.
        create_line([0,0], [1,1])"""
        return cls(org=Point(org), basis=Poly([direction]))

    def __str__(self):
        return f"<o={self.org} b={self.basis}>"

    def space_dim(self) -> int:
        """The dimensionality of the space we are part of"""
        return self.basis.dim()
    
    def my_dim(self) -> int:
        """The dimensionality of the subspace, assuming the vectors are independent"""
        return self.basis.num()
    
    @classmethod
    def from_combination(cls, comb: Combination) -> Self:
        return cls(org=comb.at(0), basis=Poly(comb.v.p[1:] - comb.v.p[0]))
    
    def as_combination(self) -> Combination:
        return Combination(np.concatenate((np.zeros((1, self.space_dim())), self.basis.p), axis=0) + self.org.c)
    
    def allclose(self, o: Self):
        return self.org.allclose(o.org) and self.basis.allclose(o.basis)
    
    def apply_to(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Return the absolute coordinates of the point(s) represented relative to this span"""
        return self.basis.apply_to(subject).add(self.org)
    
    def extract_from(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Represent the point(s) in `subject` relative to this Span"""
        return self.basis.extract_from(subject.sub(self.org))

    def extract_from_span(self, subject: Self) -> Self:
        """Represent a Span (subject) relative to this Span"""
        return self.__class__(
            org=self.extract_from(subject.org),
            basis=self.extract_from(subject.basis)
        )
    
    def get_line_point(self, d: float) -> Point:
        """Convencience function to get a point on the line represented by the Span"""
        # Of course this can be done faster
        return self.apply_to(subject=Point([d]))

    def rotate(self, coords: Iterable[int], rad: float, around_origin: bool = False) -> Self:
        new_org = self.org
        if around_origin:
            new_org = new_org.rotate(coords=coords, rad=rad)
        return self.__class__(org=new_org, basis=self.basis.rotate(coords=coords, rad=rad))
    
    def persp_reduce(self, focd: float):
        return self.__class__(
            org=self.org.persp_reduce(focd),
            basis=self.basis.persp_reduce(focd)
        )
    
    def intersect_lines(self, other: Self, test=False):
        """Return [alpha, beta] for two lines represented as
        self =  o1 + alpha * d1 (-inf<alpha<inf)
        other = o2 + beta  * d2 (-inf<beta<inf)
        where o1 + alpha * d1 = o2 + beta * d2.
        Return [None, None] if the lines are parallel.
        """
        assert self.space_dim() == 2
        assert other.space_dim() == 2
        assert self.my_dim() == 1
        assert other.my_dim() == 1
        o = other.org.c - self.org.c # [a,b]
        dir_self = self.basis.p[0] # [x0,y0]
        dir_other = other.basis.p[0] # [x1,y1]
        disc = dir_self[1] * dir_other[0] - dir_self[0] * dir_other[1] # y0*x1 - x0*y1
        if disc == 0:
            return (None, None)
        beta  = (dir_self[0] * o[1] - dir_self[1] * o[0]) / disc # (x0*b - y0*a)/disc
        alpha = (dir_other[0] * o[1] - dir_other[1] * o[0]) / disc # (x1*b - y1*a)/disc
        if test:
            assert self.get_line_point(alpha).allclose(other.get_line_point(beta))
        return (alpha, beta)
        

class Body(Span):

    def decompose(self):
        raise NotImplementedError("Implement in subclasses")
    
    def distance_on(self, line: Span) -> float:
        """Return, in multiples of alpha (where line = o + alpha * d)
        the distance of this body to o.
        Returns None if the line misses the body."""
        assert line.space_dim() == 2
        assert line.my_dim() == 1
        assert self.space_dim() == 2
        if self.my_dim() > 1:
            d = [x.distance_on(line) for x in self.decompose()]
            d = [x for x in d if x is not None]
            if len(d):
                return min(d)
            return None

        # 1D body (line segment)
        alpha, beta = line.intersect_lines(self)
        # TODO Handle when lines coincide?
        if beta is None or beta < 0 or beta > 1:
            return None
        return alpha
            
    def includes(self, point: Point) -> bool:
        """Returnes whether the body includes the point"""
        raise NotImplementedError("Implement in subclasses")
