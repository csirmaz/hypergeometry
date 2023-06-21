
import numpy as np

class Point:
    
    def __init__(self, coords):
        self.c = np.array(coords)
        
    def clone(self):
        return Point(self.c)
    
    def scale(self, x: float) -> 'Point':
        return Point(self.c * x)
    
    def add(self, p: 'Point') -> 'Point':
        assert self.dim() == p.dim()
        return Point(self.c + p.c)
        
    def sub(self, p: 'Point') -> 'Point':
        assert self.dim() == p.dim()
        return Point(self.c - p.c)
    
    def dim(self) -> int:
        """Return number of dimensions"""
        return self.c.shape[0]

    def is_zero(self) -> bool:
        return (self.c.sum() == 0)
    
    def length(self) -> float:
        return np.sqrt(np.square(self.c).sum())
    
    def norm(self) -> 'Point':
        return Point(self.c / self.length())
    
    def dot(self, p: 'Point') -> float:
        return np.dot(self.c, p.c)

    def eq(self, p: 'Point') -> bool:
        return ((self.c != p.c).sum() == 0)

    def rotate(self, coords, rad: float) -> 'Point':
        """Rotate. coords is a list of 2 coordinate indices that we rotate"""
        assert len(coords) == 2
        ca, cb = coords
        s = np.sin(rad * np.pi)
        c = np.cos(rad * np.pi)
        r = self.clone()
        r.c[ca] = c * self.c[ca] + s * self.c[cb]
        r.c[cb] = -s * self.c[ca] + c * self.c[cb]
        return r
    
    def project(self, focd: float) -> 'Point':
        """Project the point onto a subspace where the last coordinate is 0.
        focd is the distance of the focal point from the origin along this coordinate.
        """
        a = focd / (self.c[-1] - focd)
        return Point(self.c[:-1] * a)


# Unit tests
if __name__ == "__main__":

    p1 = Point([0, 0, 1])
    assert p1.scale(2).scale(.5).eq(p1)
    assert p1.dim() == 3
    assert p1.add(p1).scale(.5).eq(p1)
    p2 = p1.clone()
    p2.c[0] = 1
    assert not p1.eq(p2)
    assert p2.project(-5).dim() == 2
    
