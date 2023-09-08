from typing import Any, Iterable, Union

import hypergeometry.utils as utils
from hypergeometry.utils import XCheckError, EPSILON
from hypergeometry.utils import profiling
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span

Self = Any


class Body(Span):
    """The parent class of parallelotopes and simplices, defining the API supported by
    those classes and implementing methods that are for bodies but are independent
    of the actual type"""

    def decompose(self) -> Iterable[Self]:
        raise NotImplementedError("Implement in subclasses")

    def get_triangulated_surface(self, add_face_colors: bool = False):
        raise NotImplementedError("Implement in subclasses")

    def midpoint(self) -> Point:
        raise NotImplementedError("Implement in subclasses")

    def includes_impl(self, point: Point, permission_level: int) -> bool:
        """Returns whether the body includes the point"""
        raise NotImplementedError("Implement in subclasses")

    def intersect_line_impl(self, line: Span, permissive: bool = False) -> tuple[Union[float, None], float]:
        raise NotImplementedError("Implement in subclasses")

    def get_nondegenerate_parts(self) -> Self:
        """Yield all nondegenerate faces; decompose until we get one that is not degenerate"""
        if self.my_dim() >= 1:
            # WARNING Ensure all derivations are cached here so decomposing can be cached, too
            subj = self.get_nonzeros()
            if subj.my_dim() >= 1:
                if subj.basis.is_degenerate():
                    for face in subj.decompose():
                        for f in face.get_nondegenerate_parts():
                            yield f
                else:
                    yield subj

    def includes_sub(self, point: Point, permission_level: int, use_bounding_box: bool = True) -> bool:
        """Returns whether the body contains the point.
        Manages projected bodies as well which are potentially degenerate.
        """
        assert self.space_dim() == point.dim()
        if use_bounding_box and not self.is_in_bounds(point, permission_level=1):
            if utils.DEBUG:
                print("(Body.includes_sub) Not in bounding box")
            if utils.XCHECK and utils.throttle('xcheck_bbox'):
                # Check whether we'd be inside the original body
                if self.includes_sub(point, permission_level=permission_level, use_bounding_box=False):
                    raise XCheckError(f"Body contains point outside bounding box. body={self} point={point}")
            return False
        for face in self.get_nondegenerate_parts():
            r = face.includes_impl(point, permission_level=permission_level)
            if utils.DEBUG:
                print(f"(Body.includes_sub) face contains point: {'yes' if r else 'no'}")
            if r:
                if utils.XCHECK:
                    # Check if a line to the point intersects with the body
                    tmpf, tmperr = self.intersect_line_sub(line=Span(org=Point.zeros(point.dim()), basis=Poly([point])), permissive=True)
                    if tmpf is None or tmpf > 1.0 + EPSILON:
                        raise XCheckError("A line to a point inside a body does not intersect the body")
                return True
        return False

    def intersect_line_sub(self, line: Span, permissive: bool = False) -> tuple[Union[float, None], float]:
        """Return, in multiples of alpha (where line = O + alpha * D)
        the distance of this body to O in a lower-dimensional space.
        Returns None if the line misses the body.
        Also returns a degree of error; if there is no intersection, how much it missed.
        Manages projected bodies as well which are potentially degenerate.
        """
        assert line.my_dim() == 1
        assert line.space_dim() == self.space_dim()
        min_f = None
        max_err = 0
        for face in self.get_nondegenerate_parts():
            f, err = face.intersect_line_impl(line, permissive=permissive)
            if utils.DEBUG:
                print(f"(intersect_line_sub) intersection: {f} err={err}")
            if f is None:
                if err > max_err: max_err = err
            else:
                if min_f is None or f < min_f: min_f = f
        return min_f, max_err

    def decompose_with_normals(self) -> Iterable[tuple[Self, Point]]:
        """Return the faces with their normals pointing outwards from the body"""
        profiling('Body.decompose_with_normals')
        mid = self.midpoint()
        for face in self.decompose():
            normal = face.basis.extend_to_norm_square(permission="1").at(-1)
            facemid = face.midpoint()
            m = facemid.sub(mid).dot(normal)
            if m < 0:
                normal = normal.scale(-1)
            elif m == 0:
                raise Exception("normal perpendicular to vector to midpoint?")
            yield face, normal

