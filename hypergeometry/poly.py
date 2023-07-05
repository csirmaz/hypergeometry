
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
        self.orthonormal = None  # True of False if known to be orthonormal
        self.transpose = None  # caches to save time
        self.inverse = None  # caches to save time
        self.pseudoinverse = None

    def __str__(self):
        return "(" + ",\n".join((f"{p}" for p in self.to_points())) + ")"
    
    @classmethod
    def from_identity(cls, dim: int) -> 'Poly':
        """Create a Poly from an identity matrix"""
        r = Poly(np.identity(dim))
        r.orthonormal = True
        return r

    @classmethod
    def from_random(cls, dim: int, num: int) -> 'Poly':
        """Create a Poly from values from a uniform distribution over `[0,1)`."""
        return Poly(np.random.rand(num, dim))
    
    def reset_cache(self) -> 'Poly':
        """Remove cached calculations. Use if the matrix is mutated"""
        self.orthonormal = None
        self.transpose = None
        self.inverse = None
        self.pseudoinverse = None
        return self
    
    def _get_transpose(self) -> np.ndarray:
        if self.transpose is None:
            self.transpose = self.p.transpose()
        return self.transpose
    
    def _get_inverse(self) -> np.ndarray:
        if self.inverse is None:
            self.inverse = np.linalg.inv(self.p)
        return self.inverse

    def _get_pseudoinverse(self) -> np.ndarray:
        if self.pseudoinverse is None:
            self.pseudoinverse = np.linalg.pinv(self.p)
        return self.pseudoinverse
    
    def clone(self) -> 'Poly':
        """Returns a deep clone"""
        r = Poly(self.p)
        r.orthonormal = self.orthonormal
        return r
    
    def map(self, lmbd) -> 'Poly':
        """Generate a new Poly object using a lambda function applied to Point objects"""
        return Poly([lmbd(Point(p)) for p in self.p])
        
    def dim(self) -> int:
        """Return the number of dimensions"""
        return self.p.shape[1]
    
    def num(self) -> int:
        """Return the number of points/vectors"""
        return self.p.shape[0]
    
    def is_square(self) -> bool:
        return self.p.shape[0] == self.p.shape[1]
    
    def at(self, x: int) -> Point:
        """Return the x'th point as a Point object"""
        return Point(self.p[x])
    
    def to_points(self):
        """Separate into an array of Point objects"""
        return [Point(p) for p in self.p]
    
    def eq(self, p: 'Poly') -> bool:
        return (self.p == p.p).all()
    
    def allclose(self, p: 'Poly') -> bool:
        """Return if all values of two Poly objects are sufficiently close"""
        # Prevent broadcasting
        return self.p.shape == p.p.shape and np.allclose(self.p, p.p)
    
    def add(self, p: Point) -> 'Poly':
        """Add a point/vector to each point/vector in this Poly"""
        return Poly(self.p + p.c)
    
    def sub(self, p: Point) -> 'Poly':
        """Subtract a point/vector from each point/vector in this Poly"""
        return Poly(self.p - p.c)
    
    def mean(self) -> Point:
        return Point(np.average(self.p, axis=0))
    
    def norm(self) -> 'Poly':
        """Normalise each vector"""
        return Poly(self.p / np.sqrt(np.square(self.p).sum(axis=1, keepdims=True)))
    
    def rotate(self, coords: Iterable[int], rad: float) -> 'Poly':
        """Rotate each point. coords is a list of 2 coordinate indices that we rotate"""
        # Can do it faster by directly interacting with the matrix
        r = Poly([p.rotate(coords, rad) for p in self.to_points()])
        r.orthonormal = self.orthonormal
        return r

    def persp_reduce(self, focd: float) -> 'Poly':
        """Project the points onto a subspace where the last coordinate is 0.
        `focd` is the distance of the focal point from the origin along this coordinate.
        """
        a = focd / (focd - self.p[:,-1])
        return Poly(self.p[:,:-1] * np.expand_dims(a, axis=1))
    
    def is_orthonormal(self, force=False) -> bool:
        """Returns if the collection of vectors is an orthonormal basis (vectors are unit length and pairwise perpendicular)"""
        if self.orthonormal is not None and not force:
            return self.orthonormal
        dots = self.p @ self.p.transpose()
        identity = np.identity(self.num())
        if dots.shape == identity.shape and np.allclose(dots, identity):
            self.orthonormal = True
            return True
        self.orthonormal = False
        return False

    def make_basis(self, strict=True) -> 'Poly':
        """Transform the vectors in self into an orthonormal basis (unit-length pairwise perpendicular vectors).
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
        r = Poly(out)
        assert r.is_orthonormal()
        return r
    
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
        """
        Represent the point(s) in `subject` relative to the basis in `self`.
        
        If `self` is a square matrix, require that it be invertible (that is,
        the vectors in the basis are linearly independent and so the basis spans
        the whole space), and return x=<subj>@<basis>^-1, that is, vectors x for which
        x@<basis>=<subject>.
        
        If `self` is not square, return the coordinates which make up the projection of
        the subject onto the subspace spanned by the basis relative to it. 
        If `self` is orthonormal, use the transpose, which may be more accurate.
        Otherwise, use the pseudo-inverse of the matrix, which, however, does not
        warn if the vectors in the basis are not independent.
        """
        if self.is_square():
            si = self._get_inverse()
            # Throws exception if not invertible
        else:
            if self.is_orthonormal():
                si = self._get_transpose()
            else:
                si = self._get_pseudoinverse()
        if isinstance(subject, Point):
            return Point(subject.c @ si)
        if isinstance(subject, Poly):
            return Poly(subject.p @ si)
        raise Exception("extract_base_from: unknown type")
        
