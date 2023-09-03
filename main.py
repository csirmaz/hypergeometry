
from PIL import Image
import numpy as np
import time

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON
from hypergeometry import Point, Poly, Span, Parallelotope, Simplex, Camera, ObjectFace, Light


# Define our world

def make_tree():
    trunk_color = (.6, .5, 0)
    leaves_color = (.4, 1., .3)
    trunk_width = .2
    trunk_height = 2.5
    branch_height = 2-1
    branch_horiz = .5

    boxes = [
        # trunk
        [Parallelotope.create_box([0,0,0,-1], [trunk_width, trunk_width, trunk_width, trunk_height]), trunk_color],
        # horizontal branches
        [Parallelotope.create_box([0, 0, 0, branch_height],[-branch_horiz, trunk_width, trunk_width, trunk_width]), trunk_color],
        [Parallelotope.create_box([0, 0, 0, branch_height],[trunk_width, -branch_horiz, trunk_width, trunk_width]), trunk_color],
        [Parallelotope.create_box([0, 0, 0, branch_height], [trunk_width, trunk_width, -branch_horiz, trunk_width]), trunk_color],
        [Parallelotope.create_box([trunk_width, trunk_width, trunk_width, branch_height], [branch_horiz, -trunk_width, -trunk_width, trunk_width]), trunk_color],
        [Parallelotope.create_box([trunk_width, trunk_width, trunk_width, branch_height], [-trunk_width, branch_horiz, -trunk_width, trunk_width]), trunk_color],
        [Parallelotope.create_box([trunk_width, trunk_width, trunk_width, branch_height], [-trunk_width, -trunk_width, branch_horiz, trunk_width]), trunk_color],
    ]

    objs = []
    for box in boxes:
        objs.extend(ObjectFace.from_body(box[0], color=box[1]))

    return objs

def main():
    # list of ObjectFace objects
    OBJECTS = make_tree() # TODO
    utils.DEBUG = True # DEBUG
    OBJECTS = ObjectFace.from_body(Parallelotope.create_box([1,1,1,1],[2,2,2,2]), color=[1,1,1])
    utils.DEBUG = False # DEBUG

    CAMERAS = [
        Camera(space=Span.default_span(3), focd=-10), # 3D -> 2D
        Camera(space=Span.default_span(4), focd=-10), # 4D -> 3D
    ]

    LIGHTS = [
        Light(Point([0,-5,0,-5]))
    ]

    RANGES = [ # coordinates
        2, # 2D [-a..a] x [-a..a]
    ]

    STEPS = [
        .01, # 2D
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
    print(f"image size: {IMAGE_SIZE}")
    img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))

    def coord2pix(c):
        """Convert a coordinate to a pixel index"""
        return int((c + RANGES[0]) / STEPS[0] + .5)

    def plot(x, y, c):
        """Set the color in the image at the given coordinates"""
        img_arr[coord2pix(y), coord2pix(x), :] = c

    def bigdot(y, x, c=(1,0,0)):
        """Draw a bigger dot on the image at the given coordinates"""
        img_arr[y,x,:] = c
        img_arr[y+1,x,:] = c
        img_arr[y-1,x,:] = c
        img_arr[y,x+1,:] = c
        img_arr[y,x-1,:] = c

    def draw_error(picx, picy):
        """Mark the image for areas where errors occurred"""
        img_arr[picy, picx, :] = [1.1,0,0] if ((picx + picy) % 2) == 0 else [1.1,1.1,0]

    def get_relevant_2d(im_point_2d):
        """Get a list of objects relevant for a given 2D point"""
        objs = []
        for ix, obj in enumerate(OBJECTS_PROJ[2]):
            if obj.includes_sub(im_point_2d):
                objs.append(ix)
        return objs

    print("Processing pixels")
    start_time = time.time()
    errors = {'ray3d_intersect': 0, 'no_min_obj3': 0, 'ray4d_intersect': 0, 'no_min_obj4': 0}
    percent_done = 0
    for picy in range(IMAGE_SIZE): # pixel
        for picx in range(IMAGE_SIZE): # pixel

            # DEBUG
            if picy==280 and (picx==240 or picx==210 or picx==290):
                utils.DEBUG = True
                print("NOW")
            else:
                utils.DEBUG = False

            im_point_2d = Point([picx*STEPS[0]-RANGES[0], picy*STEPS[0]-RANGES[0]]) # image point on 2D canvas

            # First check which objects are relevant based on their 2D projection
            # TODO Why is this so slow?
            relevant_obj_ix2 = get_relevant_2d(im_point_2d)
            if len(relevant_obj_ix2) == 0:
                continue

            # Use ray tracing
            ray_3d = CAMERAS[0].ray(im_point_2d)

            if utils.DEBUG:
                print(f"\n(main) 2d image point: picy={picy} picx={picx} {im_point_2d} Relevant objects in 2d: {relevant_obj_ix2} 3d ray: {ray_3d}\n")

            # Second, get the closest object(s) among the 3D projections along the ray
            relevant_obj_ix3 = {}  # {<object index>: <distance>}
            min_dist3 = None
            for ix in relevant_obj_ix2:
                d = OBJECTS_PROJ[3][ix].intersect_line_sub(ray_3d, permissive=True)
                if d is None: # DEBUG
                    # We do expect to hit the object with the 3d ray as its 2d projection contained the 2d image point
                    # So this should not happen
                    errors['ray3d_intersect'] += 1
                    draw_error(picx, picy)
                    # Run diagnostics
                    utils.DEBUG = True
                    print(f"\n>>> ray_3d intersect inconsistency")
                    print(f"  A 3D object whose 2D projection contains the 2D image point does not intersect the 3D ray")
                    print(f"  picx={picx} picy={picy} relevant obj ix={ix}")
                    print(f"  2d calculation:")
                    print(f"    im_point_2d={im_point_2d}")
                    print(f"    obj={OBJECTS_PROJ[2][ix]}")
                    tmp = OBJECTS_PROJ[2][ix].includes_sub(im_point_2d)
                    print(f"    includes? {'yes' if tmp else 'no'}")
                    print(f"  3d calculation:")
                    print(f"    ray3d={ray_3d}")
                    print(f"    obj={OBJECTS_PROJ[3][ix]}")
                    tmp = OBJECTS_PROJ[3][ix].intersect_line_sub(ray_3d)
                    print(f"    Intersects? {'yes' if tmp else 'no'}")
                    utils.DEBUG = False
                    continue
                if utils.DEBUG:
                    print(f"\n(main) Object {ix} intersects the 3d ray at {d}\n")

                # d is a value in the context of ray_3d (a multiplier if its vector), so comparisons makes sense
                if min_dist3 is None or d < min_dist3 + EPSILON:
                    # We keep all objects whose distance is between the minimum and minimum+EPSILON
                    if min_dist3 is None:
                        assert len(relevant_obj_ix3) == 0
                        min_dist3 = d
                        relevant_obj_ix3[ix] = d
                    elif d < min_dist3:
                        min_dist3 = d
                        relevant_obj_ix3 = {ti: td for ti, td in relevant_obj_ix3.items() if td <= min_dist3 + EPSILON}
                    else:
                        # min_dist3 <= d < min_dist3 + EPSILON
                        relevant_obj_ix3[ix] = d

            if len(relevant_obj_ix3) == 0:
                errors['no_min_obj3'] += 1
                draw_error(picx, picy)
                continue

            # We now need to select the object that is closest to the 4D->3D camera's focal point
            im_point_3d = ray_3d.get_line_point(min_dist3)
            ray_4d = CAMERAS[1].ray(im_point_3d)

            if utils.DEBUG:
                print(f"\n(main) Closest objects on the 3d ray: {relevant_obj_ix3} Min dist={min_dist3} 3d image point: {im_point_3d} 4d ray: {ray_4d}\n")

            min_dist4 = None
            min_obj_ix4 = None
            for ix, dist3 in relevant_obj_ix3.items():
                obj = OBJECTS[ix]
                dist4 = obj.body.intersect_line_sub(ray_4d, permissive=True)
                if dist4 is None:
                    # We know the 3D projection of this body contains (roughly, allowing for EPSILON) im_point_3d,
                    # so we don't expect this to happen
                    errors['ray4d_intersect'] += 1
                    draw_error(picx, picy)
                    # Run diagnostics
                    utils.DEBUG = True
                    print(f"\n>>> ray_4d intersect inconsistency")
                    print(f"  An object's 3D projection intersects the 3D ray, but the 4D object doesn't intersect the 4D ray from the 3D intersection point")
                    print(f"  obj_ix={ix}")
                    print(f"  3d calculation:")
                    print(f"    obj={OBJECTS_PROJ[3][ix]}")
                    print(f"    3d ray={ray_3d} min dist={min_dist3} obj dist={dist3}")
                    print(f"    3d image point={im_point_3d}")
                    print(f"    Sanity check: Is the 3d image point inside the 3d object?")
                    tmp = OBJECTS_PROJ[3][ix].includes_sub(im_point_3d)
                    print(f"    Includes? {'yes' if tmp else 'no'}")
                    print(f"  4d calculation:")
                    print(f"    4d ray={ray_4d}")
                    print(f"    obj={OBJECTS[ix]}")
                    tmp = OBJECTS[ix].body.intersect_line_sub(ray_4d, permissive=True)
                    print(f"    Intersects? {'yes' if tmp else 'no'}")
                    utils.DEBUG = False
                    continue
                if utils.DEBUG:
                    print(f"\n(main) Object {ix} intersects the 4d ray at {dist4}\n")

                if min_dist4 is None or dist4 < min_dist4:
                    min_dist4 = dist4
                    min_obj_ix4 = ix

            if min_obj_ix4 is None:
                errors['no_min_obj4'] += 1
                draw_error(picx, picy)
                continue

            if utils.DEBUG:
                print(f"(main) Closest object in 4d: {min_obj_ix4} dist={min_dist4}")
            intersect_point_4d = ray_4d.get_line_point(min_dist4)
            img_arr[picy,picx,:] = OBJECTS[min_obj_ix4].get_color(point=intersect_point_4d, lights=LIGHTS, eye=CAMERAS[1].focal)

        percent = int(picy / IMAGE_SIZE * 100 + .5)
        if percent > percent_done:
            percent_done = percent
            spent_time = time.time() - start_time
            remaining_time = spent_time / percent_done * (100 - percent_done)
            print(f"{percent}% {remaining_time:.0f} s remaining, {errors} errors so far")

    bigdot(280,240) # DEBUG
    bigdot(280,210) # DEBUG
    bigdot(280,290) # DEBUG

    # Save the image
    # img_max = np.max(img_arr, axis=(0,1)) + 1e-7 # per channel
    img_max = np.max(img_arr) + 1e-7 # overall max
    img_data = (img_arr / img_max * 255.).astype('B')
    img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    img.save('image.png')

    print(f"Errors encountered: {errors}")

# import cProfile
# cProfile.run('main()')
main()
utils.print_profile()
