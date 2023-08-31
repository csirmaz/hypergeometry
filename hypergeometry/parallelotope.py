from typing import Any, Iterable, Union
Self=Any

from hypergeometry.utils import EPSILON, select_of, loop_many_to
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body


class Parallelotope(Body):
    """An n-dimensional parallelotope defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi <= 1.

    Note that the perspective projection of a parallelotope is not a parallelotope
    (while this applies to simplices).
    """

    @classmethod
    def create_box(cls, org, sizes) -> Self:
        """Create a box (hyperrectangle) whose edges are parallel to the axes.
        If a coordinate in `sizes` is None, it is left out, resulting in a lower-
        dimensional box.
        E.g. create_box([0,0,0], [1,None,2]) ->
            basis= 1,0,0
                   0,0,2
        """
        dim = len(org)
        assert dim == len(sizes)
        basis = [
            [(v if x == ix else 0) for x in range(dim)] 
            for ix, v in enumerate(sizes)
            if v is not None
        ]
        return cls(org=Point(org), basis=Poly(basis))
    
    def decompose(self, diagonal: bool = False) -> Iterable[Self]:
        """Return the n-1-dimensional faces.
        We don't worry about the orientation.
        """
        assert not diagonal
        if self.decomposed is not None:
            return self.decomposed
        inv = self.basis.scale(-1)
        org2 = self.org.add(self.basis.sum())
        o = ([
            Parallelotope(
                org=self.org,
                basis=self.basis.subset(indices)
            ) for indices in select_of(num=self.my_dim()-1, max=self.my_dim())
        ] + [
            Parallelotope(
                org=org2,
                basis=inv.subset(indices)
            ) for indices in select_of(num=self.my_dim()-1, max=self.my_dim())
        ])
        self.decomposed = o
        return self.decomposed
            
    def midpoint(self) -> Point:
        return self.org.add( self.basis.sum().scale(.5) )
        
    def _includes(self, point: Point) -> bool:
        """Returns whether the point is in the body"""
        # Can only be used if the body is NOT degenerate
        # (vectors in basis are independent)
        raise NotImplementedError()
        # Should be the same as Simplex.intersect_line with the exception of checking for the sum of the coordinates
    
    def _intersect_line(self, line: Span, permissive: bool = False) -> Union[float, None]:
        """Given a line represented as a Span
        (P = L0 + alpha Lv), return min(alpha) for which P falls inside
        this body, that is, the distance of this body from L0.
        Return None if there is no intersection.
        """
        raise NotImplementedError()
        # Should be the same as Simplex.intersect_line with the exception of checking for the sum of the coordinates

    def generate_grid(self, density: int) -> Iterable[Point]:
        """Yield a series of points inside the body"""
        for i in loop_many_to(num=self.my_dim(), max_=density, scaled=True):
            yield self.apply_to(Point(i))
