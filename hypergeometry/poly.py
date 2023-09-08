
from typing import Iterable, Union, Any, List, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import NP_TYPE, DETERMINANT_LIMIT, EPSILON, profiling, NotIndependentError, XCheckError
from hypergeometry.point import Point

Self = Any


class Poly:
    """A collection of points or vectors represented as a matrix"""
    
    def __init__(self, points, origin: Optional[str] = None):
        """Accepts a 2D matrix or a list of points.
        `origin` indicates what called the constructor, for debugging
        """
        if len(points) > 0 and isinstance(points[0], Point):
            points = [p.c for p in points]
        self.p = np.array(points, dtype=NP_TYPE)
        self.orthonormal = None  # True of False if known to be orthonormal
        self.independent = None # True or False if known whether the vectors are linearly independent
        self.independent_reason = None # for debugging
        self.transpose = None  # caches np.array to save time
        self.inverse = None  # caches np.array to save time
        self.pseudoinverse = None # caches np.array to save time
        self.bounds = None # caches np.array
        self.degenerate = None # True or False if known to be degenerate
        self.square = None # caches Poly() to save time
        self.norm_square = None # caches Poly() to save time
        self.nonzeros = None # caches Poly() to save time
        # profiling(f'Poly.__init__({origin})', self)

    def __str__(self):
        o = "" if self.num() <= 1 else "\n    "
        rows = [ "(" + ", ".join((f"{x:.6f}" for x in r)) + ")" for r in self.p ]
        o += "Py(" + ",\n       ".join(rows) + ")"
        return o

    @classmethod
    def from_identity(cls, dim: int) -> Self:
        """Create a Poly from an identity matrix"""
        r = cls(np.identity(dim, dtype=NP_TYPE))
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
        self.degenerate = None
        self.square = None
        self.norm_square = None
        self.nonzeros = None
        return self
    
    def clone(self) -> Self:
        """Returns a deep clone"""
        r = self.__class__(self.p, origin=f'Poly.clone[{id(self)}]')
        return r
     
    def _get_transpose(self) -> np.ndarray:
        if self.transpose is None:
            profiling('Poly.transpose:do', self)
            self.transpose = self.p.transpose()
        return self.transpose
    
    def _get_inverse(self) -> np.ndarray:
        if self.inverse is None:
            profiling('Poly.inverse:do', self)
            try:
                self.inverse = np.linalg.inv(self.p)
            except np.linalg.LinAlgError:
                self.inverse = False
        if self.inverse is False:
            raise NotIndependentError()
        return self.inverse

    def _get_pseudoinverse(self) -> np.ndarray:
        if self.pseudoinverse is None:
            profiling('Poly.pseudoinverse:do', self)
            self.pseudoinverse = np.linalg.pinv(self.p)
        else:
            profiling('Poly.pseudoinverse:cache')
        return self.pseudoinverse

    def get_bounds(self) -> np.ndarray:
        """Returns a 2-vector poly containing the min and max coordinates"""
        if self.bounds is None:
            self.bounds = np.concatenate((
                np.min(self.p, axis=0, keepdims=True),
                np.max(self.p, axis=0, keepdims=True)
            ))
        return self.bounds

    def map(self, lmbd) -> Self:
        """Generate a new Poly object using a lambda function applied to Point objects"""
        return self.__class__([lmbd(Point(p)) for p in self.p], origin='Poly.map')
         
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
        return self.__class__(self.p[indices], origin=f'Poly.subset[{id(self)}]')

    def has_zero_vec(self) -> bool:
        """Return if the poly has a zero vector"""
        return np.any(np.all(np.abs(self.p) < EPSILON, axis=1))

    def get_nonzeros(self):
        """Return the subset of vectors that are not all 0"""
        if self.nonzeros is not None:
            profiling('Poly.get_nonzeros:cache', self)
            return self.nonzeros
        profiling('Poly.get_nonzeros:do', self)
        r = self.__class__(
            self.p[
                np.any(
                    np.abs(self.p) >= EPSILON,
                    axis=1
                )
            ],
            origin=f'Poly.get_nonzeros[{id(self)}]'
        )
        assert r.dim() == self.dim()
        assert r.num() <= self.num()
        self.nonzeros = r
        return r

    def to_points(self):
        """Separate into an array of Point objects"""
        return [Point(p) for p in self.p]
    
    def eq(self, p: Self) -> bool:
        return (self.p == p.p).all()
    
    def allclose(self, p: Self) -> bool:
        """Return if all values of two Poly objects are sufficiently close"""
        # Should only be used for testing
        profiling('Poly.allclose(!)', self)
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
        return self.__class__(self.p / np.sqrt(np.square(self.p).sum(axis=1, keepdims=True)), origin=f'Poly.norm[{id(self)}]')
    
    def rotate(self, coords: List[int], rad: float) -> Self:
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
        return self.__class__(self.p[:,:-1] * np.expand_dims(a, axis=1), origin=f'Poly.persp_reduce[{id(self)}]')
    
    def is_orthonormal(self) -> bool:
        """Returns if the collection of vectors is an orthonormal basis (vectors are unit length and pairwise perpendicular)"""
        if self.orthonormal is not None:
            profiling('Poly.is_orthonormal:cache', self)
            return self.orthonormal
        profiling('Poly.is_orthonormal:do', self)
        dots = self.p @ self.p.transpose()
        identity = np.identity(self.num())
        if dots.shape == identity.shape and np.allclose(dots, identity):
            self.orthonormal = True
            return True
        self.orthonormal = False
        return False

    def is_degenerate(self) -> bool:
        """Return if vectors in this matrix are, or are almost, linearly dependent."""
        # Unlike (the inverse of) is_independent(), this function returns True
        # if a square matrix is close to being degenerate, even if technically numpy
        # can calculate an inverse (containing very large numbers). Using such an inverse
        # for mapping leads to numerical instability and nonsense results.
        if self.degenerate is not None:
            profiling('Poly.is_degenerate:cache', self)
            return self.degenerate
        profiling('Poly.is_degenerate:do', self)
        if self.num() <= 1:
            self.degenerate = False
            return self.degenerate
        if self.independent is not None and not self.independent:
            self.degenerate = True
            return self.degenerate
        if self.has_zero_vec():
            if utils.DEBUG:
                print(f"(poly:is_degenerate) Yes as has 0 vector")
            self.degenerate = True
            return self.degenerate
        subj = self
        if not self.is_square():
            # For non-square matrices, we extend first to a square matrix with vectors
            # that are perpendicular to the existing ones
            if not self.is_independent():
                if utils.DEBUG:
                    print(f"(poly:is_degenerate) degenerate as not is_independent")
                self.degenerate = True
                return self.degenerate
            if utils.DEBUG:
                print(f"(poly:is_degenerate) using extended")
            subj = self.extend_to_norm_square(permission="pos")
        d = np.linalg.det(subj.p)  # determinant
        if utils.DEBUG:
            print(f"(poly:is_degenerate) det={d}")
        self.degenerate = (abs(d) < DETERMINANT_LIMIT)
        return self.degenerate

    def is_independent(self) -> bool:
        """Returns if the rows as vectors are linearly independent"""
        # Note that this is a STRICT view of independence. Vectors in a matrix may be almost
        # linearly dependent, while technically numpy can still calculate an inverse. See
        # is_degenerate().
        if self.independent is not None:
            profiling('Poly.is_independent:cache', self)
            if utils.DEBUG:
                print(f"(is_independent) cached {self.independent} because {self.independent_reason}")
            return self.independent

        profiling('Poly.is_independent:do', self)
        if self.orthonormal:
            self.independent = True
            self.independent_reason = "orthonormal cache"
            return True
        if self.inverse is not None:
            self.independent = True
            self.independent_reason = "inverse cache"
            return True

        if self.is_square():
            try:
                self._get_inverse()
                self.independent = True
                self.independent_reason = "_get_inverse"
            except NotIndependentError:
                self.independent = False
                self.independent_reason = "not _get_inverse"
            return self.independent
        if self.num() > self.dim():
            self.independent = False
            self.independent_reason = "tall"
            return False
        try:
            self.make_basis(strict=True)
            self.independent = True
            self.independent_reason = "make_basis"
        except NotIndependentError:
            self.independent = False
            self.independent_reason = "not make_basis"
        return self.independent

    def make_basis(self, strict: bool = True) -> Self:
        """Transform the vectors in self into an orthonormal basis (unit-length pairwise perpendicular vectors).
        If strict=False, may leave out vectors if they are not linearly independent.
        """
        profiling('Poly.make_basis', self)
        if utils.DEBUG:
            print(f"(make_basis) Called with {self}")
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
        r = self.__class__(out, origin=f'Poly.make_basis[{id(self)}]')
        assert r.is_orthonormal() # DEBUG
        return r

    def extend_to_square(self, permission: str) -> Self:
        """Ensure that these vectors are linearly independent and return an extended Poly
        whose vectors are also linearly independent and has a square shape

        permission: "any" "pos" "1" -- how many extra dimensions we expect to add
        """
        if self.num() > self.dim():
            raise Exception("extend_to_square: tall matrix")
        if not self.is_independent():
            raise NotIndependentError("extend_to_square: not independent")

        if permission == "any":
            if self.is_square():
                profiling('Poly.extend_to_square:noop', self)
                return self
        else:
            if self.is_square():
                raise Exception(f"extend_to_square: already square and permission is {permission}")
            if permission == "pos":
                pass
            elif permission == "1":
                assert self.num() == self.dim() - 1
            else:
                raise Exception("extend_to_square: Unknown permission {permission}")

        if self.square is not None:
            profiling('Poly.extend_to_square:cache', self)
            return self.square

        profiling('Poly.extend_to_square:do', self)
        while True:
            e = self.__class__.from_random(dim=self.dim(), num=(self.dim() - self.num()))
            n = self.__class__(np.concatenate((self.p, e.p), axis=0), origin=f'Poly.extend_to_square[{id(self)}]')
            assert n.is_square()
            if n.is_independent():
                self.square = n
                if utils.DEBUG:
                    print(f"(poly:extend_to_square) Extended to {n}")
                return n

    def extend_to_norm_square(self, permission: str) -> Self:
        """Ensure these vectors are linearly independent and return an extended square Poly
        where the additional vectors are perpendicular to the lower dimensional original poly

        permission: "any" "pos" "1" -- how many extra dimensions we expect to add
        """
        if self.num() > self.dim():
            raise Exception("extend_to_norm_square: tall matrix")

        # Repeat here from extend_to_square as restriction cannot be cached
        if permission == "any":
            if self.is_square():
                profiling('Poly.extend_to_norm_square:noop')
                return self
        else:
            if self.is_square():
                raise Exception(f"extend_to_norm_square: already square and permission is {permission}")
            if permission == "pos":
                pass
            elif permission == "1":
                assert self.num() == self.dim() - 1
            else:
                raise Exception("extend_to_norm_square: Unknown permission {permission}")

        if self.norm_square is not None:
            profiling('Poly.extend_to_norm_square:cache', self)
            return self.norm_square
        profiling('Poly.extend_to_norm_square:do', self)
        sq = self.extend_to_square(permission=permission).make_basis()
        r = self.__class__(np.concatenate((self.p, sq.p[self.num():]), axis=0), origin=f'Poly.extend_to_norm_square[{id(self)}]')
        assert r.is_square()
        if utils.DEBUG:
            print(f"(poly:extend_to_norm_square) Extended to {r}")
        self.norm_square = r
        return r

    def apply_to(self, subject: Union['Poly', Point]) -> Union['Poly', Point]:
        """Get the linear combination of vectors in `self` according to the vector(s) in `subject`.
        If `self` is a basis, this converts vector(s) expressed in that basis into absolute coordinates."""
        profiling('Poly.apply_to', self)
        assert subject.dim() == self.num() # DIM==bNUM
        if isinstance(subject, Point):
            return subject.__class__(subject.c @ self.p, origin='Poly.apply_to') # <(1), DIM> @ <bNUM, bDIM> -> <(1), bDIM>
        if isinstance(subject, Poly):
            return subject.__class__(subject.p @ self.p, origin='Poly.apply_to') # <NUM, DIM> @ <bNUM, bDIM> -> <NUM, bDIM>
        raise Exception("apply_to: unknown type")
    
    def extract_from(self, subject: Union['Poly', Point], allow_projection: bool = False) -> Union['Poly', Point]:
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
        profiling('Poly.extract_from', self)
        if self.is_square():
            if utils.DEBUG:
                print(f"(poly:extract_from) is_square")
            si = self._get_inverse()
            # Throws exception if not invertible
            projected = False
        else:
            assert self.num() < self.dim() # Otherwise guaranteed that the vectors are not independent and so the operation doesn't make sense
            if not allow_projection:
                raise Exception("extract_from: projection is not allowed")
            projected = True
            if self.is_orthonormal():
                if utils.DEBUG:
                    print(f"(poly:extract_from) is_orthonormal")
                si = self._get_transpose()
            else:
                # Getting the pseudoinverse does not warn if the vectors are not independent
                if not self.is_independent():
                    # WARNING even if the matrix passes this filter, it may be very narrow (small determinant) making
                    # the results unstable
                    raise Exception("extract_from: not independent")
                if utils.DEBUG:
                    print(f"(poly:extract_from) get pseudoinverse")
                si = self._get_pseudoinverse()
                # TODO # Alternative idea is to use extend_to_norm_square() and invert as then we can verify the results
                # TODO # as we do below
        if utils.DEBUG:
            print(f"(poly:extract_from) self={self} si={self.__class__(si)}")

        if isinstance(subject, Point):
            r = subject.c @ si
            if utils.DEBUG:
                print(f"(poly:extract_from) result: {Point(r)}")
            if utils.XCHECK and not projected:
                # Check reverse as matrices that are close to being degenerate will not give correct result
                if not np.allclose(r @ self.p, subject.c):
                    raise XCheckError(f"extract_from: Invalid result det={np.linalg.det(self.p)}")
            return Point(r, origin='Poly.extract_from')

        if isinstance(subject, Poly):
            r = subject.p @ si
            if utils.DEBUG:
                print(f"(poly:extract_from) result: {Poly(r)}")
            if utils.XCHECK and not projected:
                # Check reverse as matrices that are close to being degenerate will not give correct result
                if not np.allclose(r @ self.p, subject.p):
                    raise XCheckError(f"extract_from: Invalid result det={np.linalg.det(self.p)}")
            return Poly(r, origin='Poly.extract_from')

        raise Exception("extract_from: unknown type")
        
