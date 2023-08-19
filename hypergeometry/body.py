from typing import Union

from hypergeometry.point import Point
from hypergeometry.span import Span


class Body(Span):
    """The parent class of parallelotopes and simplices, defining the API supported by
    those classes and implementing methods that are for bodies but are independent
    of the actual type"""

    def decompose(self):
        raise NotImplementedError("Implement in subclasses")

    def decompose_to(self, dim: int):
        """Return dim-dimensional faces of the body"""
        assert dim < self.my_dim()
        if dim == self.my_dim() - 1:
            for face in self.decompose():
                yield face
        else:
            for h in self.decompose_to(dim + 1):
                for face in h.decompose():
                    yield face

    def midpoint(self) -> Point:
        raise NotImplementedError("Implement in subclasses")

    def includes(self, point: Point) -> bool:
        """Returns whether the body includes the point"""
        raise NotImplementedError("Implement in subclasses")
    
    def includes_sub(self, point: Point) -> bool:
        """Returns whether the projection of the body contains the point"""
        assert self.space_dim() == point.dim()
        for face in self.decompose_to(self.space_dim()):
            if face.basis.is_independent():
                if face.includes(point):
                    return True
        return False

    def intersect_line_sub(self, line: Span) -> Union[float, None]:
        """Return, in multiples of alpha (where line = O + alpha * D)
        the distance of this projected body to O in a lower-dimensional space.
        Returns None if the line misses the body."""
        assert line.my_dim() == 1
        assert line.space_dim() == self.space_dim()
        # We can only calculate intersections for non-degenerate spans
        # so we try all sub-spans of all dimensions
        for target_dim in range(self.my_dim(), 0, -1):
            min_f = None
            for subspan in self.get_subspans(dim=target_dim): # TODO PROB USE DECOMOPOSE_TO & REMOVE SUBSPANS
                if subspan.basis.is_independent():
                    f = subspan.intersect_line(line)
                    if f is not None:
                        if min_f is None or f < min_f: min_f = f
            if min_f is not None:
                return min_f
        return None

    def distance_on_2d(self, line: Span) -> float:
        """Return, in multiples of alpha (where line = O + alpha * D)
        the distance of this body to O in a 2D space.
        Returns None if the line misses the body."""
        
        # TODO Extend to any dim; use util to get subset of vectors
        assert line.space_dim() == 2
        assert line.my_dim() == 1
        assert self.space_dim() == 2
        if self.my_dim() > 1:
            d = [x.distance_on_2d(line) for x in self.decompose()]
            d = [x for x in d if x is not None]
            if len(d):
                return min(d)
            return None

        # 1D body (line segment)
        alpha, beta = line.intersect_lines_2d(self)
        # TODO Handle when lines coincide?
        if beta is None or beta < 0 or beta > 1:
            return None
        return alpha
            
    def decompose_with_normals(self):
        """Return the faces with their normals pointing outwards from the body"""
        mid = self.midpoint()
        for face in self.decompose():
            normal = face.basis.extend_to_square().at(-1).norm()
            facemid = face.midpoint()
            m = facemid.sub(mid).dot(normal)
            if m < 0:
                normal = normal.scale(-1)
            elif m == 0:
                raise Exception("normal perpendicular to vector to midpoint?")
            yield face, normal
            
