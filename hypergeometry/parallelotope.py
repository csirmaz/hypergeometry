from typing import Any, Iterable, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import select_of, loop_many_to, profiling
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body
from hypergeometry.simplex import Simplex

Self = Any


class Parallelotope(Body):
    """An n-dimensional parallelotope defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi <= 1.

    Note that the perspective projection of a parallelotope is not a parallelotope
    (while this applies to simplices).
    """

    @classmethod
    def create_box(cls, org, sizes, name: Optional[str] = None) -> Self:
        """Create a box (hyperrectangle) whose edges are parallel to the axes.
        If a coordinate in `sizes` is None, it is left out, resulting in a lower-
        dimensional box.
        E.g. create_box([0,0,0], [1,None,2]) ->
            basis= 1,0,0
                   0,0,2
        """
        dim = len(org)
        assert dim == len(sizes)
        basis = [
            [(v if x == ix else 0) for x in range(dim)] 
            for ix, v in enumerate(sizes)
            if v is not None
        ]
        return cls(org=Point(org), basis=Poly(basis), name=name)
    
    def decompose(self) -> Iterable[Self]:
        """Return the n-1-dimensional faces.
        We don't worry about the orientation.
        """
        if self.decomposed is not None:
            profiling('Parallelotope.decompose:cache')
            return self.decomposed
        profiling('Parallelotope.decompose:do')
        inv = self.basis.scale(-1)
        org2 = self.org.add(self.basis.sum())
        o1 = [
            Parallelotope(
                org=self.org,
                basis=self.basis.subset(indices),
                parent=self,
                derivation_method='decompose',
                name=f"face#{i}"
            ) for i, indices in enumerate(select_of(num=self.my_dim()-1, max=self.my_dim()))
        ]
        o2 = [
            Parallelotope(
                org=org2,
                basis=inv.subset(indices),
                parent=self,
                derivation_method='decompose',
                name=f"face#{len(o1) + i}"
            ) for i, indices in enumerate(select_of(num=self.my_dim()-1, max=self.my_dim()))
        ]
        self.decomposed = o1 + o2
        return self.decomposed
            
    def midpoint(self) -> Point:
        return self.org.add( self.basis.sum().scale(.5) )
        
    def generate_grid(self, density: int) -> Iterable[Point]:
        """Yield a series of points inside the body"""
        for i in loop_many_to(num=self.my_dim(), max_=density, scaled=True):
            yield self.apply_to(Point(i))

    def split_into_simplices(self) -> Iterable[Simplex]:
        if self.my_dim() < 2:
            yield Simplex(org=self.org, basis=self.basis, parent=self, derivation_method='split1d', name='split#0')
        elif self.my_dim() == 2:
            yield Simplex(org=self.org, basis=self.basis, parent=self, derivation_method='split2d', name='split#0')
            yield Simplex(org=self.apply_to(Point([1., 1.])), basis=self.basis.scale(-1), parent=self, derivation_method='split2d', name='split#1')
        elif self.my_dim() == 3:
            # These are the vertices of the cube that are to be used for the tetrahedra that align with the edges
            #    DC hg
            #    AB ef
            split_conf = [[0,0,0],  # -> 0,0,0 | 1,0,0  0,1,0  0,0,1    A|BDe
                          [1,1,0],  # -> 1,1,0 | 0,1,0  1,0,0  1,1,1    C|DBg
                          [1,0,1],  # -> 1,0,1 | 0,0,1  1,1,1  1,0,0    f|egB
                          [0,1,1]]  # -> 0,1,1 | 1,1,1  0,0,1  0,1,0    h|geD
            for ic, conf in enumerate(split_conf):
                ubasis = np.zeros((self.my_dim(), self.my_dim()))
                for j in range(self.my_dim()):
                    ubasis[j, j] = 1 if conf[j] == 0 else -1  # Note: the basis contains the difference
                yield Simplex(
                    org=self.apply_to(Point(conf)),
                    basis=self.basis.apply_to(Poly(ubasis)),
                    parent=self,
                    derivation_method='split3d',
                    name=f"split#{ic}"
                )
            # The internal simplex: 1,1,1 | 0,0,1  0,1,0  1,0,0    g|eDB
            yield Simplex(
                org=self.apply_to(Point([1,1,1])),
                basis=self.basis.apply_to(Poly([[-1,-1,0], [-1,0,-1], [0,-1,-1]])),
                parent=self,
                derivation_method='split3d',
                name=f"split#{len(split_conf)}"
            )
        else:
            raise NotImplementedError("split_into_simplices not implemented for higher dimensions")

    def get_triangulated_surface(self, add_face_colors: bool = False):
        """Return a list of simplices covering the surface of this body"""
        if utils.DEBUG:
            print(f"(Parallelotope.get_tri) self={self} self as points={self.as_points()}")
        color = None
        for face, normal in self.decompose_with_normals():
            if add_face_colors:
                color = np.random.rand(3)
                color /= np.max(color)
            if utils.DEBUG:
                print(f"(Parallelotope.get_tri) face={face} face as points={face.as_points()}")
            for simplex in face.split_into_simplices():
                if utils.DEBUG:
                    print(f"(Parallelotope.get_tri) simplex={simplex} simplex as points={simplex.as_points()}")
                yield simplex, normal, color
