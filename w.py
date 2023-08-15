
from PIL import Image
import numpy as np

from hypergeometry import Point, Poly, Span, Combination, Parallelotope, Simplex, Camera, ObjectFace

box = Parallelotope.create_box

# Define our world

OBJECTS = []
OBJECTS.extend(ObjectFace.from_body(
    box([2,2,2,2], [1,1,1,1]).rotate([0,2], .9)
))

CAMERAS = [
    Camera(space=Span.default_span(3), focd=-10), # 3D -> 2D
    Camera(space=Span.default_span(4), focd=-10), # 4D -> 3D
]

RANGES = [ # coordinates
    3, # 2D [-a..a] x [-a..a]
    5  # 3D [0..a]
]

STEPS = [
    .005, # 2D
    .05   # 3D
]

o = OBJECTS[0].body
print(o)
for camera in CAMERAS[::-1]:
    o = camera.project(o)
    print(o)


IMAGE_SIZE = int(RANGES[0] / STEPS[0] * 2)
MAX_PICZ = int(RANGES[1] / STEPS[1])
img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

percent_done = 0       
# IDEA: start with the previous picz value
# TODO picz = 0 # We retain this value to start the search at the same place for the next 2D pixel
# IDEA: ignore empty regions
for picy in range(IMAGE_SIZE): # pixel
    for picx in range(IMAGE_SIZE): # pixel
        im_point_2d = Point([picx*STEPS[0]-RANGES[0], picy*STEPS[0]-RANGES[0]]) # image point on 2D canvas
        ray_3d = CAMERAS[0].ray(im_point_2d)
        
        min_obj = None # the object we hit
        min_dist_4d = None # the distance of the object on ray_4d
        for picz in range(MAX_PICZ):
            dist_3d = picz*STEPS[1] # The distance of the point on ray_3d from the camera focal point
            im_point_3d = ray_3d.get_line_point(dist_3d)
            ray_4d = CAMERAS[1].ray(im_point_3d)
            for o in OBJECTS:
                dist_4d = o.body.intersect_line(ray_4d)
                if dist_4d is not None and dist_4d >= 0: # We don't want things behind the 4D camera
                    if min_dist_4d is None or dist_4d < min_dist_4d:
                        min_obj = o
                        min_dist_4d = dist_4d
            if min_obj is not None:
                # We want the first ray that intersects with an object (minimum picz ~ dist_3d)
                break
        if min_obj is not None:
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
            
