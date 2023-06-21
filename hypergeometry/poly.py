
from typing import Iterable
import numpy as np

from point import Point

class Poly:
    """A collection of points represented as a matrix"""
    
    def __init__(self, points):
        """Accepts a 2D matrix or a list of points"""
        if isinstance(points[0], Point):
            points = [p.c for p in points]
        self.p = np.array(points)

    def clone(self) -> 'Poly':
        return Poly(self.p)
        
    def at(self, x: int) -> Point:
        """Return the x'th point as a Point object"""
        return Point(self.p[x])
    
    def dim(self) -> int:
        return self.at(0).dim()
    
    def to_points(self):
        return [Point(p) for p in self.p]
    
    def mean(self) -> Point:
        return Point(np.average(self.p, axis=0))
    
    def rotate(self, coords: Iterable[int], rad: float) -> 'Poly':
        # Can do it faster by directly interacting with the matrix
        return Poly([p.rotate(coords, rad) for p in self.to_points()])
    
    def is_base(self) -> bool:
        """Returns if the collection of vectors is a base (unit length and perpendicular)"""
        dots = self.p @ self.p.transpose()
        identity = np.identity(len(self.p))
        return ((dots != identity).sum() == 0)


# Unit tests
if __name__ == "__main__":

    p1 = Point([0,0,1])
    p2 = Point([0,1,0])
    p = Poly([p1, p2])
    assert p.dim() == 3
    assert p1.add(p2).scale(.5).eq(p.mean())
    assert p.is_base()
