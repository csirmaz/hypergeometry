
from typing import Union, List, Any, Optional
import numpy as np

from hypergeometry.utils import profiling, EPSILON
from hypergeometry.point import Point
from hypergeometry.poly import Poly

Self = Any


class Span:
    """This class represents a subspace as a set of points that are spanned by a set of vectors (basis)
    from a given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where -inf < xi < +inf
    """
    
    def __init__(self, org: Point, basis: Poly, origin: Optional[str] = None):
        """`origin` indicates what called the constructor, for debugging"""
        assert org.dim() == basis.dim()
        assert isinstance(org, Point)
        assert isinstance(basis, Poly)
        self.org = org
        self.basis = basis
        self.bounds = None # Cache <np.ndarray>
        self.decomposed = None # Cache of D-1-dimensional faces <List[Self]>
        self.nonzeros = None # Cache <Self>
        self.norm_square = None # Cache <Self>
        # profiling(f'Span.__init__({origin})', self)

    @classmethod
    def default_span(cls, dim: int) -> Self:
        """Convenience function to create a Span for the whole space at the origin"""
        return cls(org=Point.zeros(dim), basis=Poly.from_identity(dim))
        
    @classmethod
    def create_line(cls, org: Any, direction: Any) -> Self:
        """Convenience function to create a Span representing a line, e.g.
        create_line([0,0], [1,1])"""
        return cls(org=Point(org), basis=Poly([direction]))

    def __str__(self):
        return f"Span<org={self.org} basis={self.basis}>"

    def space_dim(self) -> int:
        """The dimensionality of the space we are part of"""
        return self.basis.dim()
    
    def my_dim(self) -> int:
        """The dimensionality of the subspace, assuming the vectors are independent"""
        return self.basis.num()

    def allclose(self, o: Self):
        # Should be used for testing only
        profiling('Span.allclose(!)')
        return self.org.allclose(o.org) and self.basis.allclose(o.basis)

    def _get_bounds(self) -> np.ndarray:
        """Get the bounding box (min/max points) for this Span"""
        if self.bounds is None:
            self.bounds = self.basis._get_bounds() + self.org.c
            self.bounds[0] -= EPSILON
            self.bounds[1] += EPSILON
        return self.bounds

    def is_in_bounds(self, p: Point) -> bool:
        """Returns whether the point is in the bounding box of this Span"""
        b = self._get_bounds()
        # This is much faster than a |...
        return np.all((p.c >= b[0]) & (p.c <= b[1]))

    def apply_to(self, subject: Union[Point, Poly, Self]) -> Any:
        """Return the absolute coordinates of the point(s) represented relative to this span"""
        if isinstance(subject, Span):
            return subject.__class__(
                org=self.apply_to(subject.org),
                basis=self.basis.apply_to(subject.basis) # We don't want to shift vectors
            )
        return self.basis.apply_to(subject).add(self.org)
    
    def extract_from(self, subject: Union[Point, Poly, Self], debug: bool = False) -> Any:
        """Represent the point(s) (not vectors!) in `subject` relative to this Span"""
        if isinstance(subject, Span):
            return subject.__class__(
                org=self.extract_from(subject.org),
                basis=self.basis.extract_from(subject.basis)  # We don't want to shift vectors
            )
        return self.basis.extract_from(subject.sub(self.org))

    def get_line_point(self, d: float) -> Point:
        """Convencience function to get a point on the line represented by the Span"""
        assert self.my_dim() == 1
        return Point(self.org.c + self.basis.p[0] * d)

    def rotate(self, coords: List[int], rad: float, around_origin: bool = False) -> Self:
        new_org = self.org
        if around_origin:
            new_org = new_org.rotate(coords=coords, rad=rad)
        return self.__class__(org=new_org, basis=self.basis.rotate(coords=coords, rad=rad), origin='Span.rotate')
    
    def persp_reduce(self, focd: float):
        profiling('Span.persp_reduce')
        org_img = self.org.persp_reduce(focd)
        return self.__class__(
            org=org_img,
            basis=self.basis.add(self.org).persp_reduce(focd).sub(org_img),
            origin=f'Span.persp_reduce[{id(self)}]'
        )
    
    def extend_to_norm_square(self, permission: str) -> Self:
        """Return a new Span whose basis is a square extension"""
        if self.norm_square is not None:
            profiling('Span.extend_to_norm_square:cache', self)
            return self.norm_square
        profiling('Span.extend_to_norm_square:do', self)
        r = self.__class__(
            org=self.org,
            basis=self.basis.extend_to_norm_square(permission=permission),
            origin=f'Span.extend_to_norm_square[{id(self)}]'
        )
        self.norm_square = r
        return r

    def get_nonzeros(self) -> Self:
        """Return the subset of vectors that are not all 0"""
        if self.nonzeros is not None:
            profiling('Span.get_nonzeros:cache', self)
            return self.nonzeros
        profiling('Span.get_nonzeros:do', self)
        r = self.__class__(
            org=self.org,
            basis=self.basis.get_nonzeros(),
            origin='Span.get_nonzeros'
        )
        self.nonzeros = r
        return r

    def intersect_lines_2d(self, other: Self, test=False):
        """Return [alpha, beta] for two lines represented as
        self =  o1 + alpha * d1 (-inf<alpha<inf)
        other = o2 + beta  * d2 (-inf<beta<inf)
        where o1 + alpha * d1 = o2 + beta * d2.
        In a 2D space.
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
        
