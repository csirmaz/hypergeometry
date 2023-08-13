from typing import Any, Iterable, Union
Self=Any
import numpy as np

from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body

class Parallelotope(Body):
    """An n-dimensional parallelotope defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi <= 1
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
        # TODO Doesn't work on projected bodies where NUM>DIM (use or(decompose))
        assert self.space_dim() == point.dim()
        p = self.extract_from(point)
        return ((p.c >= 0).all() and (p.c <= 1).all())
    
    def intersect_line(self, line: Span) -> Union[float, None]:
        """Given a line represented as a Span
        (P = L0 + alpha Lv), return min(alpha) for which P falls inside
        this body, that is, the distance of this body from L0.
        Return None if there is no intersection.
        """
        assert self.space_dim() == line.space_dim()
        assert line.my_dim() == 1
        basis_span = self.extend_to_square()
        line2 = basis_span.extract_from(line)
        all_min = None
        all_max = None
        for i in range(line2.space_dim()):
            org = line2.org.c[i]
            vec = line2.basis.p[0, i]
            # We need 0 <= org + alpha * vec <= 1 for all coordinates
            if vec == 0:
                if org < 0 or org > 1:
                    return None
            else:
                this_min = -org/vec
                this_max = (1-org)/vec
                if vec < 0:
                    this_min, this_max = this_max, this_min
                if all_min is None or all_min < this_min: all_min = this_min
                if all_max is None or all_max > this_max: all_max = this_max
        assert all_min is not None
        if all_min > all_max: return None
        return all_min
