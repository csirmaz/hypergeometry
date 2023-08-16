
from PIL import Image
import numpy as np
import time

from hypergeometry import Point, Poly, Span, Combination, Parallelotope, Simplex, Camera, ObjectFace, Light

box = Parallelotope.create_box

# Define our world

OBJECTS = []
OBJECTS.extend(ObjectFace.from_body(
    box([1,1,1,1], [1,1,1,1]).rotate([0,2], .9),
    random_color=True
))

CAMERAS = [
    Camera(space=Span.default_span(3), focd=-10), # 3D -> 2D
    Camera(space=Span.default_span(4), focd=-10), # 4D -> 3D
]

LIGHTS = [
    Light(Point([0,-5,0,-5]))
]

RANGES = [ # coordinates
    3, # 2D [-a..a] x [-a..a]
    4  # 3D [0..a]
]

STEPS = [
    .02, # 2D
    .1   # 3D
]

# o = OBJECTS[0].body
# print(o)
# for camera in CAMERAS[::-1]:
#     o = camera.project(o)
#     print(o)


IMAGE_SIZE = int(RANGES[0] / STEPS[0] * 2)
MAX_PICZ = int(RANGES[1] / STEPS[1])
print(f"image size: {IMAGE_SIZE} max picz: {MAX_PICZ}")
img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

# IDEA: Project to 2D and ignore empty regions

start_time = time.time()
percent_done = 0       
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
            # We have min_obj, min_dist_4d, picz
            intersect_point_4d = ray_4d.get_line_point(min_dist_4d)
            img_arr[picy,picx,:] = min_obj.get_color(point=intersect_point_4d, lights=LIGHTS, eye=CAMERAS[1].focal)
        
    percent = int(picy / IMAGE_SIZE * 100 + .5)
    if percent > percent_done:
        percent_done = percent
        spent_time = time.time() - start_time
        remaining_time = spent_time / percent_done * (100 - percent_done)
        print(f"{percent}% {remaining_time} s remaining")
            
# Save the image
img_max = np.max(img_arr, axis=(0,1))
img_data = (img_arr / img_max * 255.).astype('B')
img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
img.save('image.png')
            
