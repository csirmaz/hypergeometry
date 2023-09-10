from typing import Union, Optional
import numpy as np

from hypergeometry.utils import NP_TYPE, loop_many_to, EPSILON

Self = 'Point'


class Point:
    """Represents a point or vector of arbitrary dimensions"""
    
    def __init__(self, coords, derivation_method: Optional[str] = None):
        """`origin` indicates what called the constructor, for debugging"""
        self.c = np.array(coords, dtype=NP_TYPE)
    
    def __str__(self):
        return "Pt(" + ", ".join((f"{x:.6f}" for x in self.c)) + ")"
    
    @classmethod
    def zeros(cls, dim):
        """Create a Point of `dim` dimensions with all coordinates being 0"""
        return cls(np.zeros((dim), dtype=NP_TYPE))
    
    @classmethod
    def ones(cls, dim: int) -> Self:
        """Create a Point of `dim` dimensions with all coordinates being 1"""
        return cls(np.ones((dim), dtype=NP_TYPE))
    
    @classmethod
    def all_coords_to(cls, dim: int, v: float) -> Self:
        """Create a Point of `dim` dimensions with all coordinates being `v`"""
        return cls([v for i in range(dim)])
    
    @classmethod
    def generate_grid(cls, dim: int, steps: int) -> Self:
        """Yield all points in the [0,1]**dim region (inclusive) in a grid
        that has `steps+1` points along the axes."""
        for r_ in loop_many_to(num=dim, max_=steps, scaled=True):
            yield cls(r_)

    def reset_cache(self) -> Self:
        return self

    def clone(self) -> Self:
        """Return a deep clone"""
        return self.__class__(self.c)
     
    def scale(self, x: float) -> Self:
        """Scale the current vector/point by a scalar"""
        return self.__class__(self.c * x)
     
    def add(self, p: Self) -> Self:
        assert self.dim() == p.dim()
        return self.__class__(self.c + p.c)
         
    def sub(self, p: Self) -> Self:
        assert self.dim() == p.dim()
        return self.__class__(self.c - p.c)

    def dim(self) -> int:
        """Return the number of dimensions"""
        return self.c.shape[0]

    def is_zero(self) -> bool:
        """Return whether all coordinates are very close to 0"""
        return np.all(np.abs(self.c) < EPSILON)

    def length(self) -> float:
        """Return the length of the vector"""
        return np.sqrt(np.square(self.c).sum())
    
    def norm(self) -> Self:
        l = self.length()
        if l == 0:
            raise Exception("normalising 0 vector")
        return self.__class__(self.c / self.length())
    
    def dot(self, p: Self) -> float:
        return np.dot(self.c, p.c)

    def eq(self, p: Self) -> bool:
        return (self.c == p.c).all()
    
    def lt(self, p: Self) -> bool:
        return (self.c < p.c).all()

    def le(self, p: Self) -> bool:
        return (self.c <= p.c).all()

    def gt(self, p: Self) -> bool:
        return (self.c > p.c).all()

    def ge(self, p: Self) -> bool:
        return (self.c >= p.c).all()

    def allclose(self, p: Self) -> bool:
        return self.c.shape == p.c.shape and np.allclose(self.c, p.c)

    def rotate(self, coords, rad: float) -> Self:
        """Rotate. coords is a list of 2 coordinate indices that we rotate"""
        assert len(coords) == 2
        ca, cb = coords
        s = np.sin(rad * np.pi)
        c = np.cos(rad * np.pi)
        r = self.clone().reset_cache()
        r.c[ca] = c * self.c[ca] + s * self.c[cb]
        r.c[cb] = -s * self.c[ca] + c * self.c[cb]
        return r
    
    def persp_reduce(self, focd: float) -> Union[Self, None]:
        """Project the point onto a subspace where the last coordinate is 0.
        focd is the distance of the focal point from the origin along this coordinate.
        Returns None if the point is behind the focal point.
        """
        if self.c[-1] < focd + EPSILON:
            return None
        a = focd / (focd - self.c[-1])
        return self.__class__(self.c[:-1] * a)

