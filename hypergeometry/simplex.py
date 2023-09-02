from typing import Iterable, Union, Any
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON, PERMISSIVE_EPS, select_of, loop_natural_bin, profiling
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body
from hypergeometry.parallelotope import  Parallelotope

Self = Any


class Simplex(Body):
    """An n-dimensional simplex defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi and Sum(xi) <= 1
    """

    @classmethod
    def from_cube(cls, dim: int) -> Iterable[Self]:
        """Return simplices arranged on the surface of a unit hypercube, effectively "triangulating" it"""
        out = []
        for ix, a in enumerate(loop_natural_bin(dim)):
            if (ix % 2) == 0:
                basis = np.zeros((dim, dim))
                for j in range(dim):
                    basis[j, j] = 1 if a[j] == 0 else -1
                out.append(cls(org=Point(a), basis=Poly(basis)))
        return out

    @classmethod
    def from_parallelotope(cls, p: Parallelotope) -> Iterable[Self]:
        """Return simplices arranged on the surface the given parallelotope, effectively "triangulating" it"""
        return [
            p.apply_to(simplex)
            for simplex in cls.from_cube(p.my_dim())
        ]

    def decompose(self, diagonal: bool = True) -> Iterable[Self]:
        """Return the n-1-dimensional faces of this simplex.
        Note: we don't worry about the orientation"""
        if self.decomposed is not None:
            profiling('Simplex.decompose:cache', self)
            if diagonal:
                return self.decomposed
            return self.decomposed[:-1]
        profiling('Simplex.decompose:do', self)
        o = [
            Simplex(
                org=self.org,
                basis=self.basis.subset(indices),
                origin='Simplex.decompose'
            ) for indices in select_of(num=self.my_dim()-1, max=self.my_dim())
        ]
        o.append(Simplex(
            org=self.org.add(self.basis.at(0)),
            basis=Poly(self.basis.p[1:] - self.basis.p[0]),
            origin='Simplex.decompose'
        ))
        self.decomposed = o
        if diagonal:
            return self.decomposed
        return self.decomposed[:-1]

    def midpoint(self) -> Point:
        return self.org.add( self.basis.sum().scale(1./(self.my_dim() + 1.)) )
        
    def _includes(self, point: Point) -> bool:
        """Returns whether the point is in the body"""
        # Can only be used if the body is NOT degenerate
        # (vectors in basis are independent and not 0)
        profiling('Simplex.includes')
        assert self.space_dim() == point.dim()
        my_dims = self.my_dim()
        basis_span = self.extend_to_norm_square(permission="any")
        p = basis_span.extract_from(point)
        r = ((p.c >= -EPSILON).all() and ((p.c[my_dims:]) <= EPSILON).all() and np.sum(p.c) <= 1+EPSILON)
        if utils.DEBUG and r:
            print(f"(Simplex:includes) Yes, {self} includes {point}")
            print(f"(Simplex:includes) p={p} sum={np.sum(p.c)} EPSILON={EPSILON}")
        return r
    
    def _intersect_line(self, line: Span, permissive: bool = False) -> Union[float, None]:
        """Given a line represented as a Span
        (P = L0 + alpha Lv), return min(alpha) for which P falls inside
        this body, that is, the distance of this body from L0.
        Return None if there is no intersection.
        """
        profiling('Simplex.intersect_line')
        assert self.space_dim() == line.space_dim()
        assert line.my_dim() == 1
        my_dims = self.my_dim()
        basis_span = self.extend_to_norm_square(permission="any")
        line2 = basis_span.extract_from(line)
        if utils.DEBUG:
            print(f"(Simplex:intersect_line) extended span={basis_span}")
            print(f"(Simplex:intersect_line) extracted line={line2}")
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

            # Use permissive=True if we know there should be an intersection, and we're hunting for it.
            # Numeric errors can cause vec to deviate from 0 enough not to allow an intersection to be found.
            if vec == 0 or (permissive and abs(vec) < PERMISSIVE_EPS):
                if org < -EPSILON or org > mparam + EPSILON:
                    if utils.DEBUG:
                        print(f"  (Simplex:intersect_line) No intersection at i={i} as vec=0, mparam={mparam} org={org}")
                    return None
                else:
                    if utils.DEBUG:
                        print(f"  (Simplex:intersect_line) i={i}: vec=0, mparam={mparam} org={org} (OK)")
            else:
                this_min = (-EPSILON-org)/vec
                this_max = (mparam+EPSILON-org)/vec
                if vec < 0:
                    this_min, this_max = this_max, this_min
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) i={i} vec={vec} org={org} mparam={mparam} tmin={this_min} tmax={this_max}")
                if all_min is None or all_min < this_min: all_min = this_min
                if all_max is None or all_max > this_max: all_max = this_max

        # We also need (0-eps <=) SUM(org + alpha * vec) = SUM(org) + alpha * SUM(vec) <= 1+eps
        sum_org = np.sum(line2.org.c)
        sum_vec = np.sum(line2.basis.p[0])
        if sum_vec == 0:
            if sum_org < -EPSILON or sum_org > 1 + EPSILON:
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) No intersection as sum_vec=0, sum_org={sum_org}")
                return None
            else:
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) sum_vec=0, sum_org={sum_org} (OK)")
        else:
            this_min = (-EPSILON-sum_org)/sum_vec
            this_max = (1+EPSILON-sum_org)/sum_vec
            if sum_vec < 0:
                this_min, this_max = this_max, this_min
            if utils.DEBUG:
                print(f"(Simplex:intersect_line) [sum] tmin={this_min} tmax={this_max}")
            if all_min is None or all_min < this_min: all_min = this_min
            if all_max is None or all_max > this_max: all_max = this_max
        assert all_min is not None
        if all_min > all_max:
            if utils.DEBUG:
                print(f"(Simplex:intersect_line) No intersection due to combination")
            return None
        return all_min
