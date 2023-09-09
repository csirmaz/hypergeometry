from typing import Any, Iterable, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import NP_TYPE
from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.simplex import Simplex
from hypergeometry.parallelotope import Parallelotope
from hypergeometry.objectface import ObjectFace


def scatter_sphere(*,
                   org: list[float],
                   rad: float,
                   n: int,
                   size: float,
                   color: tuple[float, float, float],
                   surface: str = "translucent",
                   name: Optional[str] = None
) -> list[ObjectFace]:
    """Create a scatter of D-1-dimensional simplices in a sphere.
    The results can be used directly (as they are the same dimensions as faces)
    without triangulation.

    org: Coordinates of the center of the sphere
    rad: The size of the sphere
    n: The number of simplices
    size: The size of a simplex
    """
    space_dim = len(org)
    orgp = Point(org)
    out = []
    for i in range(n):
        while True:
            mid = Point(np.random.rand(space_dim) * 2. - 1.)  # Relative to the origin
            if mid.length() <= 1.: break
        mid = mid.scale(rad).add(orgp)
        basis = (np.random.rand(space_dim - 1, space_dim) * 2. - 1.) * size
        while True:
            basis = Poly(basis)
            if not basis.is_degenerate(): break
        # The normal will point in one or another direction randomly
        normal = basis.extend_to_norm_square(permission="1").at(-1)
        simplex = Simplex(
            org=mid,
            basis=basis
        )
        out.append(ObjectFace(
            body=simplex,
            normal=normal,
            color=color,
            surface=surface
        ))
    return out


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
    i: The index of the dimension (roughly) along which the prism extends (0-indexed)
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
