
from typing import Iterable, List, Any, Optional
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import NP_TYPE
from hypergeometry.point import Point
from hypergeometry.body import Body
from hypergeometry.light import Light

Self = Any

class ObjectFace:
    """This class represents a D-1-dimensional face of a D-dimensional object in a D-dimensional space.
    For simplicity, objects and therefore faces are all simplices (and parallelotopes).
    """
    
    def __init__(
            self,
            body: Body,
            normal: Point,
            color: tuple[float, float, float] = (1., 1., 1.),
            surface: str = 'matte',
        ):
        assert body.my_dim() == body.space_dim() - 1
        assert body.space_dim() == normal.dim()
        self.body = body
        self.color = np.array(color, dtype=NP_TYPE)
        self.normal = normal
        self.surface = surface

    def __str__(self):
        return f"{{body={self.body} normal={self.normal} bodyclass={self.body.__class__}}}"

    @classmethod
    def from_body(
            cls,
            body: Body,
            color: tuple[float, float, float] = (1., 1., 1.),
            surface: str = 'matte',
            use_face_colors: bool = False
        ) -> Iterable[Self]:
        """Generate a list of ObjectFace objects from the faces of `body`"""
        assert body.my_dim() == body.space_dim()
        for face, normal, face_color in body.get_triangulated_surface(add_face_colors=use_face_colors):
            if use_face_colors:
                color = face_color
            yield cls(body=face, normal=normal, color=color, surface=surface)

    def get_color(self, point: Point, lights: List[Light], eye: Point, ambient: float = .2):
        """Determine the color of a particular point on this ObjectFace"""
        assert len(lights) == 1
        to_light = lights[0].p.sub(point).norm()
        to_eye = eye.sub(point).norm()
        
        # Matte calculation
        m = to_light.dot(self.normal)
        if m < 0: m = 0
        return self.color * (ambient + (1 - ambient)*m)

    def rotate(self, coords: List[int], rad: float, around: Optional[Point] = None) -> Self:
        return self.__class__(
            body=self.body.rotate(coords=coords, rad=rad, around=around),
            normal=self.normal.rotate(coords=coords, rad=rad),
            color=self.color,
            surface=self.surface
        )

