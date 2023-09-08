from typing import Iterable, Union, Any, List
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON, PERMISSIVE_EPS, select_of, loop_natural_bin, profiling
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body

Self = Any


class Simplex(Body):
    """An n-dimensional simplex defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi and Sum(xi) <= 1
    """

    def decompose(self) -> Iterable[Self]:
        """Return the n-1-dimensional faces of this simplex.
        Note: we don't worry about the orientation"""
        if self.decomposed is not None:
            profiling('Simplex.decompose:cache', self)
            return self.decomposed
        profiling('Simplex.decompose:do', self)
        o = [
            Simplex(
                org=self.org,
                basis=self.basis.subset(indices),
                parent=self,
                derivation_method='decompose',
                name=f"face#{i}"
            ) for i, indices in enumerate(select_of(num=self.my_dim()-1, max=self.my_dim()))
        ]
        o.append(Simplex(
            org=self.org.add(self.basis.at(0)),
            basis=Poly(self.basis.p[1:] - self.basis.p[0]),
            parent=self,
            derivation_method='decompose',
            name=f"face#{len(o)}"
        ))
        self.decomposed = o
        return self.decomposed

    def midpoint(self) -> Point:
        return self.org.add( self.basis.sum().scale(1./(self.my_dim() + 1.)) )
        
    def includes_impl(self, point: Point, permission_level: int) -> bool:
        """Returns whether the point is in the body
        strict_level: >0 for permissive, 0 for exact, <0 for restrictive
        """
        # Can only be used if the body is NOT degenerate
        # (vectors in basis are independent and not 0)
        profiling('Simplex.includes')
        assert self.space_dim() == point.dim()
        my_dims = self.my_dim()
        basis_span = self.extend_to_norm_square(permission="any")
        p = basis_span.extract_from(point)
        r = (
                (p.c >= -EPSILON * permission_level).all()
                and ((p.c[my_dims:]) <= EPSILON * permission_level).all()
                and np.sum(p.c) <= 1 + EPSILON * permission_level
        )
        if utils.DEBUG:
            print(f"(Simplex:includes) self={self} point={point} extracted={p} sum={np.sum(p.c)} EPSILON={EPSILON * permission_level}")
            if r:
                print(f"(Simplex:includes) Yes, includes")
        return r
    
    def intersect_line_impl(self, line: Span, permissive: bool = False) -> tuple[Union[float, None], float]:
        """Given a line represented as a Span
        (P = L0 + alpha Lv), return min(alpha) for which P falls inside
        this body, that is, the distance of this body from L0.
        Return None if there is no intersection.
        Also returns a degree of error; if there is no intersection, how much it missed.

        permissive: Whether treat small numbers as effectively 0 (use when we know there's an intersection)
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
            #
            # Note: we do not include epsilon in the ranges below as then it changes the result

            if i < my_dims: # an original dimension
                mparam = 1.
            else:
                mparam = 0.

            # Use permissive=True if we know there should be an intersection, and we're hunting for it.
            # Numeric errors can cause vec to deviate from 0 enough not to allow an intersection to be found.
            if vec == 0. or (permissive and abs(vec) < PERMISSIVE_EPS):
                if org < -EPSILON:
                    if utils.DEBUG:
                        print(f"(Simplex:intersect_line) No intersection at i={i} as vec=0, mparam={mparam} org={org}")
                    return None, -org
                elif org > mparam + EPSILON:
                    if utils.DEBUG:
                        print(f"(Simplex:intersect_line) No intersection at i={i} as vec=0, mparam={mparam} org={org}")
                    return None, org - mparam
                else:
                    if utils.DEBUG:
                        print(f"(Simplex:intersect_line) i={i}: vec=0, mparam={mparam} org={org} (OK)")
            else:
                this_min = (-org)/vec
                this_max = (mparam-org)/vec
                if vec < 0.:
                    this_min, this_max = this_max, this_min
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) i={i} vec={vec} org={org} mparam={mparam} tmin={this_min} tmax={this_max}")
                if all_min is None or all_min < this_min: all_min = this_min
                if all_max is None or all_max > this_max: all_max = this_max

        # We also need (0-eps <=) SUM(org + alpha * vec) = SUM(org) + alpha * SUM(vec) <= 1+eps
        sum_org = np.sum(line2.org.c)
        sum_vec = np.sum(line2.basis.p[0])
        if sum_vec == 0 or (permissive and abs(sum_vec) < PERMISSIVE_EPS):
            if sum_org < -EPSILON:
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) No intersection as sum_vec=0, sum_org={sum_org}")
                return None, -sum_org
            elif sum_org > 1. + EPSILON:
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) No intersection as sum_vec=0, sum_org={sum_org}")
                return None, sum_org - 1.
            else:
                if utils.DEBUG:
                    print(f"(Simplex:intersect_line) sum_vec=0, sum_org={sum_org} (OK)")
        else:
            this_min = (-sum_org)/sum_vec
            this_max = (1-sum_org)/sum_vec
            if sum_vec < 0:
                this_min, this_max = this_max, this_min
            if utils.DEBUG:
                print(f"(Simplex:intersect_line) [sum] tmin={this_min} tmax={this_max}")
            if all_min is None or all_min < this_min: all_min = this_min
            if all_max is None or all_max > this_max: all_max = this_max

        assert all_min is not None
        if all_min <= all_max:
            return all_min, 0
        if all_min <= all_max + EPSILON * 2.:
            return (all_max + all_min) / 2., 0
        if utils.DEBUG:
            print(f"(Simplex:intersect_line) No intersection due to combination min={all_min} max={all_max}")
        return None, all_min - all_max

    def get_triangulated_surface(self, add_face_colors: bool = False):
        """Return a list of simplices covering the surface of this body"""
        color = None
        for face, normal in self.decompose_with_normals():
            if add_face_colors:
                color = np.random.rand(3)
                color /= np.max(color)
            yield face, normal, color