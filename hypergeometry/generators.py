from typing import Any, Iterable, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import NP_TYPE
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span
from hypergeometry.body import Body
from hypergeometry.simplex import Simplex
from hypergeometry.parallelotope import Parallelotope


def create_box(org, sizes, name: Optional[str] = None) -> Parallelotope:
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
    return Parallelotope(org=Point(org), basis=Poly(basis), name=name)


def create_prism(*,
                org: list[float],
                i: int,
                r: float,
                length: Optional[float] = None,
                dest: Optional[list[float]] = None,
                name: Optional[str] = None
) -> Parallelotope:
    """Create a prism. The dimensionality of this parallelotope is always
    the dimensions of the containing space.

    org: The coordinates of the midpoint of the base
    i: The index of the dimension (roughly) along which the prism extends
    r: The thickness of the prism (radius)
    length: Either provide length or dest. The length (height) of the prism
    dest: Either provide length or dest. The midpoint of the second base
    """
    mid1 = Point(org)
    dim_ix = i
    space_dim = len(org)
    org_point = mid1.add(Point([
        (0 if j == dim_ix else -r) for j in range(space_dim)
    ]))
    if dest is not None:
        final_vec = Point(dest).sub(mid1)
    elif length is not None:
        final_vec = Point([
            (length if j == dim_ix else 0) for j in range(space_dim)
        ])
    else:
        raise Exception("Specify either length or dest")
    basis = np.identity(space_dim, dtype=NP_TYPE) * r * 2.
    basis[i] = final_vec.c
    o = Parallelotope(org=org_point, basis=Poly(basis), name=name)
    return o
