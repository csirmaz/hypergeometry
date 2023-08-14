
from PIL import Image
import numpy as np

from hypergeometry import Point, Poly, Span, Combination, Parallelotope, Simplex, Camera

box = Parallelotope.create_box

# Define our world

OBJECTS = [
    box([2,2,2,2], [1,1,1,1]).rotate([0,2], .9)
]

CAMERAS = [
    Camera(space=Span.default_span(3), focd=-10), # 3D -> 2D
    Camera(space=Span.default_span(4), focd=-10), # 4D -> 3D
]

RANGES = [ # pixels; coordinate = range*step
    50, # 2D
    50
]

STEPS = [
    .1,
    .1
]

o = OBJECTS[0]
print(o)
for camera in CAMERAS[::-1]:
    o = camera.project(o)
    print(o)


IMAGE_SIZE = RANGES[0]
img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

for picy in range(0, RANGES[0]):
    for picx in range(0, RANGES[0]):
        im_point_2d = Point([picx*STEPS[0], picy*STEPS[1]])
        ray_3d = CAMERAS[0].ray(im_point_2d)
        for picz in range(0, RANGES[1]):
            im_point_3d = ray_3d.get_line_point(picz*STEPS[1])
            ray_4d = CAMERAS[1].ray(im_point_3d)
            
# Save the image
img_max = np.max(img_arr, axis=(0,1))
img_data = (img_arr / img_max * 255.).astype('B')
img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
img.save('image.png')
            
