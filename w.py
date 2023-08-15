
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

RANGES = [ # coordinates
    3, # 2D
    5
]

STEPS = [
    .005,
    .05
]

o = OBJECTS[0]
print(o)
for camera in CAMERAS[::-1]:
    o = camera.project(o)
    print(o)


IMAGE_SIZE = int(RANGES[0] / STEPS[0])
img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

percent_done = 0
for picy in range(IMAGE_SIZE): # pixel
    for picx in range(IMAGE_SIZE): # pixel
        im_point_2d = Point([picx*STEPS[0], picy*STEPS[0]])
        ray_3d = CAMERAS[0].ray(im_point_2d)
        # print(f"y={picy} x={picx}")
        min_int_o = None
        min_int_x = None
        for picz in range(int(RANGES[1] / STEPS[1])):
            z = picz*STEPS[1]
            im_point_3d = ray_3d.get_line_point(z)
            ray_4d = CAMERAS[1].ray(im_point_3d)
            for o in OBJECTS:
                x = o.intersect_line(ray_4d)
                if x is not None:
                    if min_int_x is None or x < min_int_x:
                        min_int_o = o
                        min_int_x = x
            if min_int_o is not None:
                # We want the first ray that intersects with an object
                break
        if min_int_o is not None:
            # TODO We have min_int_o, min_int_x, z
            img_arr[picy,picx,:] = [1,1,1]
        
    percent = int(picy / IMAGE_SIZE * 100)
    if percent > percent_done:
        percent_done = percent
        print(f"{percent}% done")
            
# Save the image
img_max = np.max(img_arr, axis=(0,1))
img_data = (img_arr / img_max * 255.).astype('B')
img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
img.save('image.png')
            
