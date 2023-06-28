
import numpy as np

from hypergeometry.utils import loop_many_to

class Point:
    """Represents a point or vector of arbitrary dimensions"""
    
    def __init__(self, coords):
        self.c = np.array(coords)
    
    def __str__(self):
        return "(" + ", ".join((f"{x:.3f}" for x in self.c)) + ")"
    
    @classmethod
    def zeros(cls, dim):
        """Create a Point of `dim` dimensions with all coordinates being 0"""
        return Point(np.zeros((dim)))
    
    @classmethod
    def ones(cls, dim: int) -> 'Point':
        """Create a Point of `dim` dimensions with all coordinates being 1"""
        return Point(np.ones((dim)))
    
    @classmethod
    def all_coords_to(cls, dim: int, v: float) -> 'Point':
        """Create a Point of `dim` dimensions with all coordinates being `v`"""
        return Point([v for i in range(dim)])
    
    @classmethod
    def generate_grid(cls, dim: int, steps: int) -> 'Point':
        """Yield all points in the [0,1]**dim region (inclusive) in a grid
        that has `steps+1` points along the axes.
        WARNING Mutates and yields the same object"""
        r = cls.zeros(dim)
        for r_ in loop_many_to(num=dim, max=steps, arr=r.c, scaled=True):
            yield r
    
    def clone(self) -> 'Point':
        """Return a deep clone"""
        return Point(self.c)
    
    def scale(self, x: float) -> 'Point':
        """Scale the current vector/point by a scalar"""
        return Point(self.c * x)
    
    def add(self, p: 'Point') -> 'Point':
        assert self.dim() == p.dim()
        return Point(self.c + p.c)
        
    def sub(self, p: 'Point') -> 'Point':
        assert self.dim() == p.dim()
        return Point(self.c - p.c)
    
    def dim(self) -> int:
        """Return the number of dimensions"""
        return self.c.shape[0]

    def is_zero(self) -> bool:
        """Return all coordinates are very close to 0"""
        return self.allclose(Point.zeros(self.dim()))
    
    def length(self) -> float:
        """Return the length of the vector"""
        return np.sqrt(np.square(self.c).sum())
    
    def norm(self) -> 'Point':
        return Point(self.c / self.length())
    
    def dot(self, p: 'Point') -> float:
        return np.dot(self.c, p.c)

    def eq(self, p: 'Point') -> bool:
        return (self.c == p.c).all()
    
    def lt(self, p: 'Point') -> bool:
        return (self.c < p.c).all()

    def le(self, p: 'Point') -> bool:
        return (self.c <= p.c).all()

    def gt(self, p: 'Point') -> bool:
        return (self.c > p.c).all()

    def ge(self, p: 'Point') -> bool:
        return (self.c >= p.c).all()

    def allclose(self, p: 'Point') -> bool:
        return self.c.shape == p.c.shape and np.allclose(self.c, p.c)

    def rotate(self, coords, rad: float) -> 'Point':
        """Rotate. coords is a list of 2 coordinate indices that we rotate"""
        assert len(coords) == 2
        ca, cb = coords
        s = np.sin(rad * np.pi)
        c = np.cos(rad * np.pi)
        r = self.clone()
        r.c[ca] = c * self.c[ca] + s * self.c[cb]
        r.c[cb] = -s * self.c[ca] + c * self.c[cb]
        return r
    
    def project(self, focd: float) -> 'Point':
        """Project the point onto a subspace where the last coordinate is 0.
        focd is the distance of the focal point from the origin along this coordinate.
        """
        a = focd / (focd - self.c[-1])
        return Point(self.c[:-1] * a)

