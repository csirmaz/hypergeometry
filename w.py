
from PIL import Image
import numpy as np
import time

from hypergeometry import Point, Poly, Span, Parallelotope, Simplex, Camera, ObjectFace, Light

# Define our world

# list of ObjectFace objects
OBJECTS = ObjectFace.from_triangulated(
    #Span.default_span(4).rotate([0,2], .9),
    Parallelotope.create_box([0,0,0,-1], [.6,1,1,2]).rotate([0,2], .9),
    color=(.6, .5, 0)
)

CAMERAS = [
    Camera(space=Span.default_span(3), focd=-10), # 3D -> 2D
    Camera(space=Span.default_span(4), focd=-10), # 4D -> 3D
]

LIGHTS = [
    Light(Point([0,-5,0,-5]))
]

RANGES = [ # coordinates
    2, # 2D [-a..a] x [-a..a]
    3  # 3D [0..a]
]

STEPS = [
    .01, # 2D
    .05   # 3D
]

# Span/Body objects (so we don't have to calculate normals)
OBJECTS_PROJ = {x:[] for x in range(OBJECTS[0].body.space_dim())}
for obj in OBJECTS:
    obj = obj.body
    dim = obj.space_dim()
    for camera in CAMERAS[::-1]:
        obj = camera.project(obj)
        dim -= 1
        OBJECTS_PROJ[dim].append(obj)

IMAGE_SIZE = int(RANGES[0] / STEPS[0] * 2)
MAX_PICZ = int(RANGES[1] / STEPS[1])
print(f"image size: {IMAGE_SIZE} max picz: {MAX_PICZ}")
img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

def coord2pix(c):
    """Convert a coordinate to a pixel index"""
    return int((c + RANGES[0]) / STEPS[0] + .5)


def plot(x, y, c):
    """Set the color in the image at the given coordinates"""
    img_arr[coord2pix(y), coord2pix(x), :] = c


def bigdot(cx, cy, c):
    """Draw a bigger dot on the image at the given coordinates"""
    x = coord2pix(cx)
    y = coord2pix(cy)
    img_arr[y,x,:] = c
    img_arr[y+1,x,:] = c
    img_arr[y-1,x,:] = c
    img_arr[y,x+1,:] = c
    img_arr[y,x-1,:] = c


def draw_error(picx, picy):
    """Mark the image for areas where errors occurred"""
    img_arr[picy, picx, :] = [1.1,0,0] if ((picx + picy) % 2) == 0 else [1.1,1.1,0]


start_time = time.time()
errors = {'ray3d_intersect': 0, 'no_min_obj': 0, 'ray4d_intersect': 0}
percent_done = 0       
for picy in range(IMAGE_SIZE): # pixel
    for picx in range(IMAGE_SIZE): # pixel

        im_point_2d = Point([picx*STEPS[0]-RANGES[0], picy*STEPS[0]-RANGES[0]]) # image point on 2D canvas

        # First check which objects are relevant based on their 2D projection
        relevant_obj_ix2 = []
        for ix, obj2d in enumerate(OBJECTS_PROJ[2]):
            if obj2d.includes_sub(im_point_2d):
                relevant_obj_ix2.append(ix)

        if len(relevant_obj_ix2) == 0:
            continue

        # Use ray tracing
        ray_3d = CAMERAS[0].ray(im_point_2d)

        # Second, get the closest object among the 3D projections along the ray
        relevant_obj_ix3 = []
        min_obj_ix3 = None
        min_dist3 = None
        for ix in relevant_obj_ix2:
            d = OBJECTS_PROJ[3][ix].intersect_line_sub(ray_3d)
            if d is None: # DEBUG
                errors['ray3d_intersect'] += 1
                draw_error(picx, picy)
                # Run diagnostics
                print(f"\n>>> ray_3d intersect inconsistency")
                print(f"  An object whose projection contains the 2D image point does not intersect the 3D ray")
                print(f"  picx={picx} picy={picy} relevant obj ix={ix}")
                print(f"  2d calculation:")
                print(f"    im_point_2d={im_point_2d}")
                print(f"    obj={OBJECTS_PROJ[2][ix]}")
                tmp = OBJECTS_PROJ[2][ix].includes_sub(im_point_2d, debug=True)
                print(f"    includes? {'yes' if tmp else 'no'}")
                print(f"  3d calculation:")
                print(f"    ray3d={ray_3d}")
                print(f"    obj={OBJECTS_PROJ[3][ix]}")
                tmp = OBJECTS_PROJ[3][ix].intersect_line_sub(ray_3d, debug=True)
                continue
            # d is a value in the context of ray_3d, so this comparison makes sense
            if min_dist3 is None or d < min_dist3:
                min_dist3 = d
                min_obj_ix3 = ix

        if min_obj_ix3 is None:
            errors['no_min_obj'] += 1
            draw_error(picx, picy)
            continue
        min_obj = OBJECTS[min_obj_ix3]
        im_point_3d = ray_3d.get_line_point(min_dist3)
        ray_4d = CAMERAS[1].ray(im_point_3d)
        dist_4d = min_obj.body.intersect_line_sub(ray_4d)
        if dist_4d is None:
            errors['ray4d_intersect'] += 1
            draw_error(picx, picy)
            continue
        intersect_point_4d = ray_4d.get_line_point(dist_4d)
        img_arr[picy,picx,:] = min_obj.get_color(point=intersect_point_4d, lights=LIGHTS, eye=CAMERAS[1].focal)

    percent = int(picy / IMAGE_SIZE * 100 + .5)
    if percent > percent_done:
        percent_done = percent
        spent_time = time.time() - start_time
        remaining_time = spent_time / percent_done * (100 - percent_done)
        print(f"{percent}% {remaining_time:.0f} s remaining, {errors} errors so far")

# Save the image
# img_max = np.max(img_arr, axis=(0,1)) + 1e-7 # per channel
img_max = np.max(img_arr) + 1e-7 # overall max
img_data = (img_arr / img_max * 255.).astype('B')
img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
img.save('image.png')

print(f"Errors encountered: {errors}")
