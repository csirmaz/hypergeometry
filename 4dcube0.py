
# Draw a perspective projection of a 4D translucent cube on a 2D plane.
# This script uses a simple forward projection, where many points in the 4D
# cube are projected to the picture plane and their density is used to determine the color.
# The script can be used to project higher dimensional cubes as well.

import numpy as np
from PIL import Image
from hypergeometry import Point, Poly, Span, loop_natural_bin

DIM = 4 # Number of dimensions
CUBE_STEP = 100000 # How many random points to take in a step
FOCAL_DIST = -10 # The focal distance to use for every projection
ZOOM = 0.4
IMAGE_SIZE = 400

# Define a span for the cube
cube = Span(org=Point.all_coords_to(dim=DIM, v=-.5), basis=Poly.from_identity(DIM))

# Rotate the cube
for i in range(DIM):
    for j in range(i+1, DIM):
        r = .3 # np.random.rand() * 2.
        cube = cube.rotate((i,j), r, around_origin=True)

img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

def to_bucket(x):
    return int(x*ZOOM*IMAGE_SIZE+IMAGE_SIZE/2+.5)

# Take a number of random points in the cube and project them
while True:
    cube_points = Poly.from_random(num=CUBE_STEP, dim=DIM)
    
    real_points = cube.apply_to(cube_points)
    projection = real_points
    for i in range(DIM - 2):
        projection = projection.persp_reduce(FOCAL_DIST)
    assert projection.dim() == 2
    for i in range(CUBE_STEP):
        img_arr[to_bucket(projection.p[i,0]), IMAGE_SIZE - to_bucket(projection.p[i,1]), 0] += 1
        
    # Create points on the edges
    if False:
        edge_points = (cube_points.p * 2.).astype(int).astype(float) # Move to vertex
        r_ix = (np.random.rand(CUBE_STEP) * DIM).astype(int) # Where to keep the original coordinate?
        edge_points[range(CUBE_STEP), r_ix] = cube_points.p[range(CUBE_STEP), r_ix]
        projection = cube.apply_to(Poly(edge_points))
        for i in range(DIM - 2):
            projection = projection.presp_reduce(FOCAL_DIST)
        assert projection.dim() == 2
        for i in range(CUBE_STEP):
            img_arr[to_bucket(projection.p[i,0]), IMAGE_SIZE - to_bucket(projection.p[i,1]), 1] += 1

    # Calculate and save the image
    img_max = np.max(img_arr, axis=(0,1))
    img_data = (img_arr / img_max * 200. + (img_arr > 0) * 55.).astype('B')
    img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    img.save('image.png')
    print("Image updated", flush=True)