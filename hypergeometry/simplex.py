from typing import Iterable, Union
import numpy as np

from hypergeometry.point import Point
from hypergeometry.span import Span
from hypergeometry.body import Body

class Simplex(Body):
    """An n-dimensional simplex defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi and Sum(xi) <= 1
    """
    
    def decompose(self) -> Iterable['Simplex']:
        """Return the n-1-dimensional faces
        Note: we don't worry about the orientation"""
        o = [
            Simplex(
                org=self.org,
                basis=self.basis.except_for(i)
            ) for i in range(self.my_dim())
        ]
        o.append(Simplex(
            org=self.org.add(self.basis.at(0)),
            basis=Poly(self.basis.p[1:] - self.basis.p[0])
        ))
        return o

    def midpoint(self) -> Point:
        return self.org.add( self.basis.sum().scale(1./(self.my_dim() + 1.)) )
        
    def includes(self, point: Point) -> bool:
        """Returns whether the point is in the body"""
        # TODO Doesn't work on projected bodies where NUM>DIM (use or(decompose))
        assert self.space_dim() == point.dim()
        p = self.extract_from(point)
        return ((p.c >= 0).all() and np.sum(p.c) <= 1)
    
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
            # We need 0 <= org + alpha * vec (<= 1) for all coordinates
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
        # We also need (0 <=) SUM(org + alpha * vec) = SUM(org) + alpha * SUM(vec) <= 1
        sum_org = np.sum(line2.org.c)
        sum_vec = np.sum(line2.basis.p[0])
        if sum_vec == 0:
            if sum_org < 0 or sum_org > 1:
                return None
        else:
            this_min = -sum_org/sum_vec
            this_max = (1-sum_org)/sum_vec
            if sum_vec < 0:
                this_min, this_max = this_max, this_min
            if all_min is None or all_min < this_min: all_min = this_min
            if all_max is None or all_max > this_max: all_max = this_max
        assert all_min is not None
        if all_min > all_max: return None
        return all_min
