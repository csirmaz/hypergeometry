
# Draw a perspective projection of a 4D translucent cube on a 2D plane.
# Here there are two focal points, and a point on the picture plane corresponds
# not to a line (as in 3D), but a plane.
# The color on the picture is determined by the area of the intersection of this
# plane and the cube.

from hypergeometry import Point, Poly, Span

# The picture plane is the set of points (*, *, picture_z, picture_w)
picture_z = -5
picture_w = -5

foc1 = Point([0,0,0,-10])
foc2 = Point([0,0,-10,0])

