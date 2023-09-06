
from typing import Union, List, Any, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import profiling, EPSILON, BOUNDINGBOX_EPS, XCheckError
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
            self.bounds[0] -= BOUNDINGBOX_EPS
            self.bounds[1] += BOUNDINGBOX_EPS
        return self.bounds

    def is_in_bounds(self, p: Point, permission_level: int) -> bool:
        """Returns whether the point is in the bounding box of this Span"""
        # This is always an approximation, so we should always be permissive.
        # We also add an epsilon to the bouns in _get_bounds() for speed
        assert permission_level == 1
        b = self._get_bounds()
        if utils.DEBUG:
            print(f"(Span.is_in_bounds) point={p} box={Poly(b)}")
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
    
    def extract_from(self, subject: Union[Point, Poly, Self], allow_projection: bool = False) -> Any:
        """Represent the point(s) (not vectors!) in `subject` relative to this Span"""
        if isinstance(subject, Span):
            return subject.__class__(
                org=self.extract_from(subject.org, allow_projection=allow_projection),
                basis=self.basis.extract_from(subject.basis, allow_projection=allow_projection)  # We don't want to shift vectors
            )
        return self.basis.extract_from(subject.sub(self.org), allow_projection=allow_projection)

    def get_line_point(self, dist: float) -> Point:
        """Convenience function to get a point on the line represented by the Span"""
        assert self.my_dim() == 1
        p = Point(self.org.c + self.basis.p[0] * dist)
        if utils.XCHECK:
            s = self.extract_from(p, allow_projection=True)
            if s.dim() != 1 or abs(s.c[0] - dist) > EPSILON:
                raise XCheckError(f"get_line_point reverse calculation failed. org dist={dist} extracted={s}")
        return p

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
