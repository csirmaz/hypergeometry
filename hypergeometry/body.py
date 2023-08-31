from typing import Any, Iterable, Union
Self = Any

from hypergeometry.point import Point
from hypergeometry.span import Span


class Body(Span):
    """The parent class of parallelotopes and simplices, defining the API supported by
    those classes and implementing methods that are for bodies but are independent
    of the actual type"""

    def decompose(self, diagonal: bool = True) -> Iterable[Self]:
        raise NotImplementedError("Implement in subclasses")

    def decompose_to(self, dim: int):
        """Return dim-dimensional faces of the body"""
        assert dim <= self.my_dim()
        if dim == self.my_dim():
            yield self
        elif dim == self.my_dim() - 1:
            for face in self.decompose():
                yield face
        else:
            for h in self.decompose_to(dim + 1):
                for face in h.decompose():
                    yield face

    def midpoint(self) -> Point:
        raise NotImplementedError("Implement in subclasses")

    def includes(self, point: Point, debug: bool = False) -> bool:
        """Returns whether the body includes the point"""
        raise NotImplementedError("Implement in subclasses")
    
    def includes_sub(self, point: Point, debug: bool = False) -> bool:
        """Returns whether the projection of the body contains the point"""
        assert self.space_dim() == point.dim()
        for face in self.decompose_to(self.space_dim()):
            if debug:
                print(f"(includes_sub) face={face}")
            if face.basis.is_degenerate(debug=debug):
                if debug:
                    print(f"(includes_sub) face is degenerate")
            else:
                r = face.includes(point, debug=debug)
                if debug:
                    print(f"(includes_sub) face contains point: {'yes' if r else 'no'}")
                if r:
                    return True
        return False

    def intersect_line_sub(self, line: Span, permissive: bool = False, debug: bool = False) -> Union[float, None]:
        """Return, in multiples of alpha (where line = O + alpha * D)
        the distance of this projected body to O in a lower-dimensional space.
        Returns None if the line misses the body."""
        assert line.my_dim() == 1
        assert line.space_dim() == self.space_dim()
        # We can only calculate intersections for non-degenerate spans,
        # so we try all sub-spans of all dimensions
        for target_dim in range(
            min(self.my_dim(), self.space_dim()),  # no point in going higher than space-dim as that'll always be degenerate; we cannot decompose into a dimension more than the original
            max(0, self.space_dim() - 2),  # anything lower than D-1 will not intersect with a line (loop stops ABOVE D-2)
            -1
        ):
            min_f = None
            for face in self.decompose_to(dim=target_dim):
                if debug:
                    print(f"(intersect_line_sub) tdim={target_dim} face={face}")
                if face.basis.is_degenerate():
                    if debug:
                        print(f"(intersect_line_sub) face is degenerate")
                else:
                    f = face.intersect_line(line, permissive=permissive, debug=debug)
                    if debug:
                        print(f"(intersect_line_sub) intersection: {f}")
                    if f is not None:
                        if min_f is None or f < min_f: min_f = f
            if min_f is not None:
                return min_f
        return None

    def decompose_with_normals(self, diagonal: bool = True) -> Iterable[Self]:
        """Return the faces with their normals pointing outwards from the body"""
        mid = self.midpoint()
        for face in self.decompose(diagonal=diagonal):
            normal = face.basis.extend_to_norm_square().at(-1)
            facemid = face.midpoint()
            m = facemid.sub(mid).dot(normal)
            if m < 0:
                normal = normal.scale(-1)
            elif m == 0:
                raise Exception("normal perpendicular to vector to midpoint?")
            yield face, normal

