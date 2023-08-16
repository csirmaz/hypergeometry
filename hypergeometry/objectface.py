
from typing import Iterable, Union
import numpy as np

from hypergeometry.point import Point
from hypergeometry.simplex import Simplex
from hypergeometry.parallelotope import Parallelotope
from hypergeometry.light import Light

class ObjectFace:
    """This class represents a D-1-dimensional face of a D-dimensional object in a D-dimensional space.
    For simplicity, objects and therefore faces are all simplices and parallelotopes.
    """
    
    def __init__(
            self,
            body: Union[Simplex, Parallelotope],
            normal: Point,
            color = [1,1,1],
            surface = 'matte',
        ):
        assert body.my_dim() == body.space_dim() - 1
        assert body.space_dim() == normal.dim()
        self.body = body
        self.color = np.array(color, dtype='float')
        self.normal = normal
        
    @classmethod
    def from_body(
            cls,
            body: Union[Simplex, Parallelotope],
            color = [1,1,1],
            surface = 'matte',
            random_color = False,
        ):
        """Generate a list of pbjects from the faces of `body`"""
        assert body.my_dim() == body.space_dim()
        if random_color:
            return [cls(body=b, normal=n, color=np.random.rand(3), surface=surface) for b, n in body.decompose_with_normals()]
        return [cls(body=b, normal=n, color=color, surface=surface) for b, n in body.decompose_with_normals()]
    
    def get_color(self, point: Point, lights: Iterable[Light], eye: Point, ambient: float = .2):
        assert len(lights) == 1
        to_light = lights[0].p.sub(point).norm()
        to_eye = eye.sub(point).norm()
        
        # Matte calculation
        m = to_light.dot(self.normal)
        if m < 0: m = 0
        return self.color * (ambient + (1 - ambient)*m)
        
