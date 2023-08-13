from typing import Any, Union
Self=Any

from hypergeometry.point import Point
from hypergeometry.poly import Poly
from hypergeometry.span import Span


class Camera:
    """A camera is defined by a Span spanning the whole space with an orthonormal basis.
    The first D-1 vectors form the image pane, while the focal point is focd away
    along the last vector"""
    
    def __init__(self, space: Span, focd: float) -> Self:
        assert space.basis.is_square()
        assert space.basis.is_orthonormal()
        self.space = space
        self.focd = focd
        self.image_pane = Span(org=space.org, basis=space.basis.except_for(-1))
        self.image_dim = self.image_pane.my_dim()
        assert self.image_dim == space.my_dim() - 1
        self.focal = space.apply_to(Point([0] * self.image_dim + [focd])) # focal point
        
        
    def ray(self, p: Point) -> Span:
        """Return a line (Span) from the focal point towards point p in the image pane"""
        assert p.dim() == self.image_dim
        ipoint = self.image_pane.apply_to(p) # point on the image pane
        return Span(org=self.focal, basis=Poly([ipoint.sub(self.focal)]))
    
    def project(self, p: Union[Point, Poly, Span]) -> Any:
        """Project a point or body in the outer space onto the image pane.
        Return coordinates relative to the image pane"""
        if isinstance(p, Point) or isinstance(p, Poly) or isinstance(p, Span):
            p = self.space.extract_from(p)
            return p.persp_reduce(self.focd)
        else:
            raise Exception("unknown type")
    
    
