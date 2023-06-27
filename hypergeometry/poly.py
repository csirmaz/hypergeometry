
from typing import Iterable, Union
import numpy as np

from hypergeometry.point import Point

class Poly:
    """A collection of points or vectors represented as a matrix"""
    
    def __init__(self, points):
        """Accepts a 2D matrix or a list of points"""
        if isinstance(points[0], Point):
            points = [p.c for p in points]
        self.p = np.array(points)

    def __str__(self):
        return "(" + ",\n".join((f"{p}" for p in self.to_points())) + ")"
    
    def clone(self) -> 'Poly':
        return Poly(self.p)
        
    def at(self, x: int) -> Point:
        """Return the x'th point as a Point object"""
        return Point(self.p[x])
    
    def dim(self) -> int:
        """The number of dimensions"""
        return self.p.shape[1]
    
    def num(self) -> int:
        """The number of points/vectors"""
        return self.p.shape[0]
    
    def eq(self, p: 'Poly') -> bool:
        return ((self.p != p.p).sum() == 0)
    
    def allclose(self, p: 'Poly') -> bool:
        return np.allclose(self.p, p.p)
    
    def to_points(self):
        """Separate into an array of Point objects"""
        return [Point(p) for p in self.p]
    
    def add(self, p: Point) -> 'Poly':
        return Poly(self.p + p.c)
    
    def sub(self, p: Point) -> 'Poly':
        return Poly(self.p - p.c)
    
    def mean(self) -> Point:
        return Point(np.average(self.p, axis=0))
    
    def norm(self) -> 'Poly':
        """Normalise each vector"""
        return Poly(self.p / np.sqrt(np.square(self.p).sum(axis=1, keepdims=True)))
    
    def rotate(self, coords: Iterable[int], rad: float) -> 'Poly':
        """Rotate each point. coords is a list of 2 coordinate indices that we rotate"""
        # Can do it faster by directly interacting with the matrix
        return Poly([p.rotate(coords, rad) for p in self.to_points()])
    
    def is_norm_basis(self) -> bool:
        """Returns if the collection of vectors is a "normal" basis (vectors are unit length and pairwise perpendicular)"""
        dots = self.p @ self.p.transpose()
        identity = np.identity(self.num())
        return np.allclose(dots, identity)

    def make_basis(self, strict=True) -> 'Poly':
        """Transform the vectors in self into a "normal" basis (unit-length pairwise perpendicular vectors).
        If strict=False, may leave out vectors if they are not linearly independent.
        """
        out = [self.at(0).norm()]
        for i in range(1, self.num()):
            v = self.at(i)
            for j in range(0, i):
                d = out[j].dot(v)
                v = v.sub(out[j].scale(d))
                # print(f"i={i} j={j} v_org={self.at(i)} out[j]={out[j]} d={d} v={v} {v.is_zero()}")
                if v.is_zero():
                    if strict:
                        raise Exception("make_basis: not independent")
                    v = None
            if v is not None:
                out.append(v.norm())
        return Poly(out)
    
    def apply_to(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Get the linear combination of vectors in `self` according to the vector(s) in `subject`.
        If `self` is a basis, this converts vector(s) expressed in that basis into absolute coordinates."""
        assert subject.dim() == self.num() # DIM==bNUM
        if isinstance(subject, Point):
            return Point(subject.c @ self.p) # <(1), DIM> @ <bNUM, bDIM> -> <(1), bDIM>
        if isinstance(subject, Poly):
            return Poly(subject.p @ self.p) # <NUM, DIM> @ <bNUM, bDIM> -> <NUM, bDIM>
        raise Exception("apply_to: unknown type")
    
    def extract_from(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Return the vector(s) x for which xA=s, where A are the vectors in self, and s is the subject.
        `self` must be an invertible matrix.
        """
        si = np.linalg.inv(self.p)
        if isinstance(subject, Point):
            return Point(subject.c @ si)
        if isinstance(subject, Poly):
            return Poly(subject.p @ si)
        raise Exception("extract_base_from: unknown type")
        

# Unit tests
if __name__ == "__main__":

    p1 = Point([0,0,1])
    p2 = Point([0,1,0])
    p = Poly([p1, p2])
    assert p.dim() == 3
    assert p1.add(p2).scale(.5).eq(p.mean())
    assert p.is_norm_basis()
    assert not Poly([[3,4],[6,8]]).norm().is_norm_basis()
    assert Poly([p1.scale(2), p1.add(p2)]).make_basis().eq(p)
    assert p.apply_to(Point([1,2])).eq(Point([0,2,1]))
    assert p.apply_to(Poly([[1,2],[3,4]])).eq(Poly([[0,2,1],[0,4,3]]))

    base2 = Poly(np.identity(3)).rotate((0,1),.2).rotate((1,2),.3)
    points = Poly([[50,60,70],[-1,-3,-2]])
    assert base2.apply_to(base2.extract_from(points)).allclose(points)
    assert base2.apply_to(base2.extract_from(points.at(0))).allclose(points.at(0))
    
    assert Poly([[1,0],[1,1]]).extract_from(Point([10.9, 31.4])).allclose(Point([-20.5, 31.4]))
