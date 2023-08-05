from typing import Iterable

from hypergeometry.point import Point
from hypergeometry.span import Body

class Parallelotope(Body):
    """An n-dimensional parallelotope defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi <= 1
    """
    
    def decompose(self) -> Iterable['Parallelotope']:
        """Return the n-1-dimensional faces.
        We don't worry about the orientation.
        """
        inv = self.basis.scale(-1)
        org2 = self.org.add(self.basis.sum())
        return ([
            Parallelotope(
                org=self.org,
                basis=self.basis.except_for(i)
            ) for i in range(self.my_dim())
        ] + [
            Parallelotope(
                org=org2,
                basis=inv.except_for(i)
            ) for i in range(self.my_dim())
        ])
        
    def includes(self, point: Point) -> bool:
        """Returns whether the point is in the body"""
        assert self.space_dim() == point.dim()
        p = self.extract_from(point)
        return ((p.c >= 0).all() and (p.c <= 1).all())

