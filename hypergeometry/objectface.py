
from typing import Iterable, List
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import NP_TYPE
from hypergeometry.point import Point
from hypergeometry.body import Body
from hypergeometry.parallelotope import Parallelotope
from hypergeometry.simplex import Simplex
from hypergeometry.light import Light


class ObjectFace:
    """This class represents a D-1-dimensional face of a D-dimensional object in a D-dimensional space.
    For simplicity, objects and therefore faces are all simplices (and parallelotopes).
    """
    
    def __init__(
            self,
            body: Body,
            normal: Point,
            color = (1,1,1),
            surface = 'matte',
        ):
        assert body.my_dim() == body.space_dim() - 1
        assert body.space_dim() == normal.dim()
        self.body = body
        self.color = np.array(color, dtype=NP_TYPE)
        self.normal = normal

    def __str__(self):
        return f"{{body={self.body} normal={self.normal} bodyclass={self.body.__class__}}}"

    @classmethod
    def from_body(
            cls,
            body: Body,
            color = (1,1,1),
            surface: str = 'matte',
            diagonal: bool = True
        ):
        """Generate a list of ObjectFace objects from the faces of `body`"""
        assert body.my_dim() == body.space_dim()
        if isinstance(body, Parallelotope):
            print("WARNING: You probably don't want to use parallelotope faces as their projections are not parallelotopes.")
            print("Use from_triangulated() instead.")
        return [cls(body=b, normal=n, color=color, surface=surface) for b, n in body.decompose_with_normals(diagonal=diagonal)]

    @classmethod
    def from_triangulated(
            cls,
            body: Parallelotope,
            color=(1, 1, 1),
            surface: str = 'matte'
        ):
        """Generate a list of ObjectFace objects from a parallelotope by first "triangulating" it into simplices,
        and then taking the D-1-dimensional faces of them"""
        out = []
        if utils.DEBUG:
            print(f"(ObjectFace.from_triangulated) Triangulating body {body}")
        for simplex in Simplex.from_parallelotope(body):
            faces = cls.from_body(body=simplex, color=color, surface=surface, diagonal=False)
            if utils.DEBUG:
                print(f"(ObjectFace.from_triangulated) Simplex {simplex}; its faces become objects #{len(out)}..#{len(out)+len(faces)-1}")
                for i, f in enumerate(faces):
                    print(f"(ObjectFace.from_triangulated) Simplex face, obj #{len(out)+i}: {f}")
            out.extend(faces)
        return out

    def get_color(self, point: Point, lights: List[Light], eye: Point, ambient: float = .2):
        assert len(lights) == 1
        to_light = lights[0].p.sub(point).norm()
        to_eye = eye.sub(point).norm()
        
        # Matte calculation
        m = to_light.dot(self.normal)
        if m < 0: m = 0
        return self.color * (ambient + (1 - ambient)*m)
        
