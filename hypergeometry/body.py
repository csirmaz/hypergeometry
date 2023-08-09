
from hypergeometry.point import Point
from hypergeometry.span import Span


class Body(Span):

    def decompose(self):
        raise NotImplementedError("Implement in subclasses")
    
    def distance_on_2d(self, line: Span) -> float:
        """Return, in multiples of alpha (where line = o + alpha * d)
        the distance of this body to o in a 2D space.
        Returns None if the line misses the body."""
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
            
    def includes(self, point: Point) -> bool:
        """Returnes whether the body includes the point"""
        raise NotImplementedError("Implement in subclasses")
