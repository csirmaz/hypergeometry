
from typing import Union
import numpy as np

from hypergeometry.simplex import Simplex
from hypergeometry.parallelotope import Parallelotope

class ObjectFace:
    """This class represents a D-1-dimensional face of a D-dimensional object in a D-dimensional space.
    For simplicity, objects and therefore faces are all simplices and parallelotopes.
    """
    
    def __init__(
            self,
            body: Union[Simplex, Parallelotope],
            color = [1,1,1],
            surface = 'matte',
        ):
        assert body.my_dim() == body.space_dim() - 1
        self.body = body
        self.color = np.array(color, dtype='float')
        # Calculate the normal
        self.normal = body.basis.extend_to_square().at(-1)
        
    @classmethod
    def from_body(
            cls,
            body: Union[Simplex, Parallelotope],
            color = [1,1,1],
            surface = 'matte',
        ):
        """Generate a list of pbjects from the faces of `body`"""
        assert body.my_dim() == body.space_dim()
        return [cls(body=b, color=color, surface=surface) for b in body.decompose()]
