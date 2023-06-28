
# Draw a perspective projection of a 4D translucent cube on a 2D plane.
# Here there are two focal points, and a point on the picture plane corresponds
# not to a line (as in 3D), but a plane.
# The color on the picture is determined by the area of the intersection of this
# plane and the cube.

from hypergeometry import Point, Poly, Span, loop_natural_bin

# The picture plane is the set of points (*, *, picture_z, picture_w)
picture_x = [-20, 20, 100] # Defines from-to and number of dots
picture_y = [-20, 20, 100] # Defines from-to and number of dots
picture_z = 0
picture_w = 0

foc1 = Point([0,0,0,-10])
foc2 = Point([0,0,-10,0])

# Define the cube and remember important points
cube_points = []
direction_points = [] # Indices. These are the points along the coordinate axes (before rotation)
for c in loop_natural_bin(4):
    c_on = sum(c) # The number of 1 coordinates
    i = len(cube_points)
    if c_on == 0: assert i == 0
    if c_on == 1: direction_points.append(i)
    cube_points.append(c)
cube = Poly(cube_points).add(Point([-.5,-.5,-.5,-.5]))
# TODO Rotate the cube

# Define a span from the cube
assert len(direction_points) == 4
cube_span = Span(org=cube.at(0), basis=Poly([cube.at(x).sub(cube.at(0)) for x in direction_points]))
assert cube_span.basis.is_norm_basis()

# Loop on the canvas/picture
for i in range(picture_x[2]):
    for j in range(picture_y[2]):
        p_x = (picture_x[1] - picture_x[0]) / picture_x[2] * i + picture_x[0]
        p_y = (picture_y[1] - picture_y[0]) / picture_y[2] * j + picture_y[0]
        picture_point = Point([p_x, p_y, picture_z, picture_w])
        
        # The projection plane associated with the given point on the picture plane
        # is the one that goes through picture_point, foc1 and foc2
        basis = Poly([foc1.sub(picture_point), foc2.sub(picture_point)])
        try:
            basis = basis.make_basis()
        except Exception as e:
            print(f"Skipping picture point {picture_point} as plane is degenerate basis={basis} error={e}")
            continue
        project_plane = Span(org=picture_point, basis=basis)

