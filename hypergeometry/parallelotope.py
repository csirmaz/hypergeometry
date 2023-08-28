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
        
    def includes(self, point: Point, debug=False) -> bool:
        """Returns whether the point is in the body"""
        # Can only be used if the body is NOT degenerate
        # (vectors in basis are independent)
        assert self.space_dim() == point.dim()
        p = self.extract_from(point)
        return ((p.c >= -EPSILON).all() and (p.c <= 1 + EPSILON).all())
    
    def intersect_line(self, line: Span) -> Union[float, None]:
        """Given a line represented as a Span
        (P = L0 + alpha Lv), return min(alpha) for which P falls inside
        this body, that is, the distance of this body from L0.
        Return None if there is no intersection.
        """
        assert self.space_dim() == line.space_dim()
        assert line.my_dim() == 1
        my_dims = self.my_dim()
        basis_span = self.extend_to_square()
        print(f"  basis_span={basis_span}") # DEBUG
        line2 = basis_span.extract_from(line)
        print(f"  line2={line2}") # DEBUG
        all_min = None
        all_max = None
        for i in range(line2.space_dim()):
            org = line2.org.c[i]
            vec = line2.basis.p[0, i]
            # We need 0-eps <= org + alpha * vec <= 1+eps for all original coordinates
            # and 0-eps <= org + alpha * vec <= 0+eps for additional coordinates
            #
            # 0-eps <= org + alpha * vec <= M+eps
            # 0-eps-org <= alpha*vec <= M+eps-org
            # IF vec == 0:
            #     -eps-org <= 0 <= M+eps-org
            #     Miss if -eps-org > 0 OR 0 > M+eps-org
            #          if -eps > org OR org > M+eps
            # IF vec > 0:
            #     (-eps-org)/vec <= alpha   (this is the minimum permissible value)
            #     alpha <= (M+eps-org)/vec  (this is the maximum permissible value)
            # IF vec < 0:
            #     (-eps-org)/vec >= alpha   (this is the maximum permissible value)
            #     alpha >= (M+eps-org)      (this is the minimum permissible value)

            if i < my_dims: # an original dimension
                mparam = 1.
            else:
                mparam = 0.
            
            if vec == 0:
                if org < -EPSILON or org > 1 + EPSILON:
                    return None
            else:
                this_min = (-EPSILON-org)/vec
                this_max = (mparam+EPSILON-org)/vec
                if vec < 0:
                    this_min, this_max = this_max, this_min
                if all_min is None or all_min < this_min: all_min = this_min
                if all_max is None or all_max > this_max: all_max = this_max

        assert all_min is not None
        if all_min > all_max: return None
        return all_min

    def generate_grid(self, density: int) -> Iterable[Point]:
        """Yield a series of points inside the body"""
        for i in loop_many_to(num=self.my_dim(), max_=density, scaled=True):
            yield self.apply_to(Point(i))
