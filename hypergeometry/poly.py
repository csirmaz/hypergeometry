
from typing import Iterable, Union, Any
Self=Any
import numpy as np

from hypergeometry.point import Point

class Poly:
    """A collection of points or vectors represented as a matrix"""
    
    def __init__(self, points):
        """Accepts a 2D matrix or a list of points"""
        if isinstance(points[0], Point):
            points = [p.c for p in points]
        self.p = np.array(points, dtype='float')
        self.orthonormal = None  # True of False if known to be orthonormal
        self.independent = None # True or False if known whether the vectors are linearly independent
        self.transpose = None  # caches np.array to save time
        self.inverse = None  # caches np.array to save time
        self.pseudoinverse = None # caches np.array to save time
        self.determinant = None # caches the determinant
        self.square = None # caches Poly to save time

    def __str__(self):
        return "(" + ",\n".join((f"{p}" for p in self.to_points())) + ")"
    
    @classmethod
    def from_identity(cls, dim: int) -> Self:
        """Create a Poly from an identity matrix"""
        r = cls(np.identity(dim, dtype='float'))
        r.orthonormal = True
        return r

    @classmethod
    def from_random(cls, dim: int, num: int) -> Self:
        """Create a Poly from values from a uniform distribution over `[0,1)`."""
        return cls(np.random.rand(num, dim))
    
    def reset_cache(self) -> Self:
        """Remove cached calculations. Use if the matrix is mutated"""
        self.orthonormal = None
        self.independent = None
        self.transpose = None
        self.inverse = None
        self.pseudoinverse = None
        self.determinant = None
        self.square = None
        return self
    
    def clone(self) -> Self:
        """Returns a deep clone"""
        r = self.__class__(self.p)
        r.orthonormal = self.orthonormal
        r.independent = self.independent
        r.square = self.square
        return r
     
    def _get_transpose(self) -> np.ndarray:
        if self.transpose is None:
            self.transpose = self.p.transpose()
        return self.transpose
    
    def _get_inverse(self) -> np.ndarray:
        if self.inverse is None:
            try:
                self.inverse = np.linalg.inv(self.p)
            except np.linalg.LinAlgError:
                self.inverse = False
        if self.inverse is False:
            raise NotIndependentError()
        return self.inverse

    def _get_pseudoinverse(self) -> np.ndarray:
        if self.pseudoinverse is None:
            self.pseudoinverse = np.linalg.pinv(self.p)
        return self.pseudoinverse

    def _get_determinant(self) -> float:
        if self.determinant is None:
            self.determinant = np.linalg.det(self.p)
        return self.determinant

    def map(self, lmbd) -> Self:
        """Generate a new Poly object using a lambda function applied to Point objects"""
        return self.__class__([lmbd(Point(p)) for p in self.p])
         
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
        
    def pop(self) -> Self:
        """Return a Poly that does not contain the last point/vector"""
        return self.__class__(self.p[:-1])
    
    def subset(self, indices: Iterable[int]) -> Self:
        """Return a Poly formed from the vectors at the given indices"""
        return self.__class__(self.p[indices])
    
    def to_points(self):
        """Separate into an array of Point objects"""
        return [Point(p) for p in self.p]
    
    def eq(self, p: Self) -> bool:
        return (self.p == p.p).all()
    
    def allclose(self, p: Self) -> bool:
        """Return if all values of two Poly objects are sufficiently close"""
        # Prevent broadcasting
        return self.p.shape == p.p.shape and np.allclose(self.p, p.p)
    
    def add(self, p: Point) -> Self:
        """Add a point/vector to each point/vector in this Poly"""
        return self.__class__(self.p + p.c)
    
    def sub(self, p: Point) -> Self:
        """Subtract a point/vector from each point/vector in this Poly"""
        return self.__class__(self.p - p.c)
        
    def scale(self, x: float) -> Self:
        return self.__class__(self.p * x)
    
    def mean(self) -> Point:
        return Point(np.average(self.p, axis=0))
        
    def sum(self) -> Point:
        return Point(np.sum(self.p, axis=0))
    
    def norm(self) -> Self:
        """Normalise each vector"""
        return self.__class__(self.p / np.sqrt(np.square(self.p).sum(axis=1, keepdims=True)))
    
    def rotate(self, coords: Iterable[int], rad: float) -> Self:
        """Rotate each point. coords is a list of 2 coordinate indices that we rotate"""
        assert len(coords) == 2
        ca, cb = coords
        s = np.sin(rad * np.pi)
        c = np.cos(rad * np.pi)
        r = self.clone().reset_cache()
        r.orthonormal = self.orthonormal
        r.p[:,ca] = c * self.p[:,ca] + s * self.p[:,cb]
        r.p[:,cb] = -s * self.p[:,ca] + c * self.p[:,cb]
        return r

    def persp_reduce(self, focd: float) -> Self:
        """Project the points onto a subspace where the last coordinate is 0.
        `focd` is the distance of the focal point from the origin along this coordinate.
        """
        a = focd / (focd - self.p[:,-1])
        return self.__class__(self.p[:,:-1] * np.expand_dims(a, axis=1))
    
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

    def is_degenerate(self, debug: bool = False) -> bool:
        """Return if vectors in this matrix are, or are almost, linearly dependent."""
        # Unlike (the inverse of) is_independent(), this function returns True
        # if a square matrix is close to being degenerate, even if technically numpy
        # can calculate an inverse (containing very large numbers). Using such an inverse
        # for mapping leads to numerical instability and nonsense results.
        if self.independent is not None and not self.independent:
            return True
        if self.is_square():
            d = self._get_determinant()
            if debug:
                print(f"(poly:is_degenerate) det={d}")
            return (abs(d) < 1e-17)
        print("Warning: we currently cannot assess non-square matrices being almost-degenerate")
        return (not self.is_independent())
    
    def is_independent(self, force=False) -> bool:
        """Returns if the rows as vectors are linearly independent"""
        # Note that this is a STRICT view of independence. Vectors in a matrix may be almost
        # linearly dependent, while technically numpy can still calculate an inverse. See
        # is_degenerate().
        if self.independent is not None and not force:
            return self.independent
        if not force:
            if self.orthonormal:
                self.independent = True
                return True
            if self.inverse is not None:
                self.independent = True
                return True
        if self.is_square():
            try:
                self._get_inverse()
                self.independent = True
            except NotIndependentError:
                self.independent = False
            return self.independent
        if self.num() > self.dim():
            self.independent = False
            return False
        try:
            self.make_basis(strict=True)
            self.independent = True
        except NotIndependentError:
            self.independent = False
        return self.independent

    def make_basis(self, strict=True) -> Self:
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
                        raise NotIndependentError("make_basis: not independent")
                    v = None
            if v is not None:
                out.append(v.norm())
        r = self.__class__(out)
        assert r.is_orthonormal() # DEBUG
        return r

    def extend_to_square(self, force=False) -> Self:
        """Ensure that these vectors are linearly independent and return an extended Poly
        whose vectors are also linearly independent and has a square shape"""
        if not self.is_independent():
            raise NotIndependentError("extend_to_square: not independent")
        if self.is_square():
            return self
        if self.num() > self.dim():
            raise Exception("extend_to_square: tall matrix")
        if not force and self.square is not None:
            return self.square
        while True:
            e = self.__class__.from_random(dim=self.dim(), num=(self.dim() - self.num()))
            n = self.__class__(np.concatenate((self.p, e.p), axis=0))
            assert n.is_square()
            if n.is_independent():
                self.square = n
                return n
    
    def apply_to(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Get the linear combination of vectors in `self` according to the vector(s) in `subject`.
        If `self` is a basis, this converts vector(s) expressed in that basis into absolute coordinates."""
        assert subject.dim() == self.num() # DIM==bNUM
        if isinstance(subject, Point):
            return subject.__class__(subject.c @ self.p) # <(1), DIM> @ <bNUM, bDIM> -> <(1), bDIM>
        if isinstance(subject, Poly):
            return subject.__class__(subject.p @ self.p) # <NUM, DIM> @ <bNUM, bDIM> -> <NUM, bDIM>
        raise Exception("apply_to: unknown type")
    
    def extract_from(
            self,
            subject: Union['Poly', Point],
            allow_projection: bool = False,
            check_result: bool = False,
            debug: bool = False
    ) -> Union['Poly', Point]:
        """
        Represent the point(s) in `subject` relative to the basis in `self`.
        
        If `self` is a square matrix, require that it be invertible (that is,
        the vectors in the basis are linearly independent and so the basis spans
        the whole space), and return x=<subj>@<basis>^-1, that is, vectors x for which
        x@<basis>=<subject>.
        
        If `self` is not square and allow_projection is True, return the coordinates
        which make up the projection of
        the subject onto the subspace spanned by the basis relative to it. 
        If `self` is orthonormal, use the transpose, which may be more accurate.
        Otherwise, use the pseudo-inverse of the matrix.
        """
        if self.is_square():
            if debug:
                print(f"      (poly:extract_from) is_square")
            si = self._get_inverse()
            # Throws exception if not invertible
            projected = False
        else:
            assert self.num() < self.dim() # Otherwise guaranteed that the vectors are not independent and so the operation doesn't make sense
            if not allow_projection:
                raise Exception("extract_from: projection is not allowed")
            projected = True
            if self.is_orthonormal():
                if debug:
                    print(f"      (poly:extract_from) is_ortho")
                si = self._get_transpose()
            else:
                # Getting the pseudoinverse does not warn if the vectors are not independent
                if not self.is_independent():
                    # WARNING even if the matrix passes this filter, it may be very narrow (small determinant) making
                    # the results unstable
                    raise Exception("extract_from: not independent")
                if debug:
                    print(f"      (poly:extract_from) get pseudoinverse")
                si = self._get_pseudoinverse()
        if debug:
            print(f"      (poly:extract_from) self={self} si={si}")
        if isinstance(subject, Point):
            r = subject.c @ si
            if debug:
                print(f"      (poly:extract_from) result: Point({r})")
            if check_result and not projected:
                # Check reverse as matrices that are close to being degenerate will not give correct result
                if not np.allclose(r @ self.p, subject.c):
                    raise Exception(f"extract_from: Invalid result det={np.linalg.det(self.p)}")
            return Point(r)
        if isinstance(subject, Poly):
            r = subject.p @ si
            if debug:
                print(f"      (poly:extract_from) result: Poly({r})")
            if check_result and not projected:
                # Check reverse as matrices that are close to being degenerate will not give correct result
                if not np.allclose(r @ self.p, subject.p):
                    raise Exception(f"extract_from: Invalid result det={np.linalg.det(self.p)}")
            return Poly(r)
        raise Exception("extract_from: unknown type")
        

class NotIndependentError(Exception):
    pass
