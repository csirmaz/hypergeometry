
from PIL import Image
import numpy as np
import random

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON, XCheckError
from hypergeometry import Point, Poly, Span, Parallelotope, Simplex, Camera, ObjectFace, Light


class Renderer:

    def __init__(self,
        objects: list[ObjectFace],
        cameras: list[Camera],
        lights: list[Light],
        img_range: float,
        img_step: float
    ):
        self.objects = objects
        self.cameras = cameras
        self.lights = lights
        self.img_range = img_range # 2D coords [-a..a] x [-a..a]
        self.img_step = img_step
        self.errors = {'ray3d_intersect': 0, 'no_min_obj3': 0, 'ray4d_intersect': 0, 'no_min_obj4': 0, 'multiple_min_obj4': 0}
        self.objects_proj = None  # dict[<dimension>: list[Span]]

        self.image_size = int(self.img_range / self.img_step * 2.)
        print(f"Image size: {self.image_size}")
        self.img_arr = np.zeros((self.image_size, self.image_size, 3))

        self.prepare_projections()

    def prepare_projections(self, dump_objects: bool = False):
        self.objects_proj = {x: [] for x in range(self.objects[0].body.space_dim())}
        for ix, obj in enumerate(self.objects):
            bdy = obj.body
            dim = bdy.space_dim()
            if dump_objects:
                print(f"Object #{ix} dim={dim}: {obj}")
            for camera in self.cameras[::-1]:
                bdy = camera.project(bdy)
                dim -= 1
                self.objects_proj[dim].append(bdy)
                if dump_objects:
                    print(f"  Projection of #{ix} to dim={dim}: {bdy}")

    def get_relevant_2d(self, im_point_2d: Point) -> list[int]:
        """Get a list of object indices relevant for a given 2D point"""
        objs = []
        for ix, obj in enumerate(self.objects_proj[2]):
            if obj.includes_sub(im_point_2d, permission_level=0):
                objs.append(ix)
        return objs

    def diagnose_pixel_process(self, *,
        picy: int,
        picx: int,
        obj_ix: int,
        im_point_2d: Point,
        ray_3d: Span,
        min_dist3: float = None,
        im_point_3d: Point = None,
        ray_4d: Span = None,
    ):
        """Log and verify data and calculations related to processing an image pixel"""
        utils.debug_push()
        print(f"  picx={picx} picy={picy} relevant obj ix={obj_ix}")

        print(f"2d calculation:")
        print(f"    im_point_2d={im_point_2d}")
        obj2 = self.objects_proj[2][obj_ix]
        print(f"    2d obj={obj2}")
        print(f"    Sanity check: does the 2d projection contain the 2d image point?")
        tmp = obj2.includes_sub(im_point_2d, permission_level=0)
        print(f"    Includes? {'yes' if tmp else 'no'}")
        print(f"3d calculation:")
        print(f"    ray3d={ray_3d}")
        obj3 = self.objects_proj[3][obj_ix]
        print(f"    3d obj={obj3}")
        print(f"    Get the intersection between ray3d and the 3d obj")
        tmp, tmperr = obj3.intersect_line_sub(ray_3d, permissive=True)
        print(f"    Intersects? {f'No, err={tmperr}' if tmp is None else f'Yes at {tmp}'}")

        print(f"    3d ray={ray_3d} min dist={min_dist3} obj dist={dist3}")
        print(f"    3d image point={im_point_3d}")
        print(f"    Sanity check: Is the 3d image point inside the 3d object?")
        tmp = self.objects_proj[3][ix].includes_sub(im_point_3d, permission_level=1)
        print(f"    Includes? {'yes' if tmp else 'no'}")

        print(f"4d calculation:")
        if min_dist3 is not None: print(f"   min_dist3={min_dist3}")
        if im_point_3d is not None: print(f"    im_point_3d={im_point_3d}")
        obj4 = self.objects[obj_ix]
        print(f"    obj={obj4}")
        if ray_4d is not None:
            print(f"    4d ray={ray_4d}")
            tmp, tmperr = obj4.body.intersect_line_sub(ray_4d, permissive=True)
            print(f"    Intersects? {f'No, err={tmperr}' if tmp is None else f'Yes at {tmp}'}")
        utils.debug_pop()

    def process_img_pixel(self, picy: int, picx: int):
        im_point_2d = Point([picx * self.img_step - self.img_range, picy * self.img_step - self.img_range])  # image point on 2D canvas

        # First check which objects are relevant based on their 2D projection
        relevant_obj_ix2 = self.get_relevant_2d(im_point_2d)
        if len(relevant_obj_ix2) == 0:
            return

        # Use ray tracing
        ray_3d = self.cameras[0].ray(im_point_2d)

        if utils.DEBUG:
            print(f"\n(main) 2d image point: picy={picy} picx={picx} {im_point_2d} Relevant objects in 2d: {relevant_obj_ix2} 3d ray: {ray_3d}\n")

        # Second, get the closest object(s) among the 3D projections along the ray
        relevant_obj_ix3 = {}  # {<object index>: <distance>}
        min_dist3 = None
        for ix in relevant_obj_ix2:
            if utils.XCHECK:
                obj2 = self.objects_proj[2][ix]
                if not obj2.includes_sub(im_point_2d, permission_level=0):
                    raise XCheckError("obj2 does not contain im_point_2d")

            obj3 = self.objects_proj[3][ix]
            dist3, intersect_err3 = obj3.intersect_line_sub(ray_3d, permissive=True)
            if dist3 is None:
                # We do expect to hit the object with the 3d ray as its 2d projection contained the 2d image point
                # So this should not happen
                # IDEA: But if it does happen, don't worry about it if intersect_err3 is small
                self.errors['ray3d_intersect'] += 1
                self.draw_error(picx, picy)
                print(f"\n>>> ray_3d intersect inconsistency, err={intersect_err3}")
                self.diagnose_pixel_process(picy=picy, picx=picx, obj_ix=ix, im_point_2d=im_point_2d, ray_3d=ray_3d)
                continue

            if utils.DEBUG:
                print(f"\n(main) Object #{ix} intersects the 3d ray at dist={dist3} point={ray_3d.get_line_point(dist3)}\n")
            if utils.XCHECK:
                tmp_point = ray_3d.get_line_point(dist3)
                if not obj3.includes_sub(tmp_point, permission_level=1):
                    raise XCheckError("obj3 does not contain im_point_3d")
                tmp_point = ray_3d.get_line_point(dist3 - .1)
                if not obj3.includes_sub(tmp_point, permission_level=0):
                    raise XCheckError("obj3 contains a point closer than im_point_3d")

            # d is a value in the context of ray_3d (a multiplier if its vector), so comparisons makes sense
            if min_dist3 is None or dist3 < min_dist3 + EPSILON:
                # We keep all objects whose distance is between the minimum and minimum+EPSILON
                if min_dist3 is None:
                    assert len(relevant_obj_ix3) == 0
                    min_dist3 = dist3
                    relevant_obj_ix3[ix] = dist3
                elif dist3 < min_dist3:
                    min_dist3 = dist3
                    relevant_obj_ix3 = {ti: td for ti, td in relevant_obj_ix3.items() if td <= min_dist3 + EPSILON}
                    relevant_obj_ix3[ix] = dist3
                else:
                    # min_dist3 <= d < min_dist3 + EPSILON
                    relevant_obj_ix3[ix] = dist3

        if len(relevant_obj_ix3) == 0:
            self.errors['no_min_obj3'] += 1
            self.draw_error(picx, picy)
            return

        # We now need to select the object that is closest to the 4D->3D camera's focal point
        im_point_3d = ray_3d.get_line_point(min_dist3)
        ray_4d = self.cameras[1].ray(im_point_3d)

        if utils.DEBUG:
            print(f"\n(main) Closest objects on the 3d ray: {relevant_obj_ix3} Min dist={min_dist3} 3d image point: {im_point_3d} 4d ray: {ray_4d}\n")

        relevant_obj_ix4 = {}
        min_dist4 = None
        for ix, dist3 in relevant_obj_ix3.items():
            obj4 = self.objects[ix]
            dist4, intersect_err4 = obj4.body.intersect_line_sub(ray_4d, permissive=True)
            if dist4 is None:
                # We know the 3D projection of this body contains (roughly, allowing for EPSILON) im_point_3d,
                # so we don't expect this to happen.
                # IDEA: But if it does happen, don't worry about it if intersect_err4 is small
                self.errors['ray4d_intersect'] += 1
                self.draw_error(picx, picy)
                print(f"\n>>> ray_4d intersect inconsistency err={intersect_err4}")
                self.diagnose_pixel_process(picy=picy, picx=picx, obj_ix=ix, im_point_2d=im_point_2d, ray_3d=ray_3d,
                                            min_dist3=min_dist3, im_point_3d=im_point_3d, ray_4d=ray_4d)
                continue

            if utils.DEBUG:
                print(f"\n(main) Object {ix} intersects the 4d ray at dist={dist4} point={ray_4d.get_line_point(dist4)}\n")
            if utils.XCHECK:
                tmp_point = ray_4d.get_line_point(dist4)
                if not obj4.body.includes_sub(tmp_point, permission_level=1):
                    raise XCheckError("obj4 does not contain point_4d")
                tmp_point = ray_4d.get_line_point(dist4 - .1)
                if not obj4.body.includes_sub(tmp_point, permission_level=0):
                    raise XCheckError("obj4 contains a point closer than point_4d")

            if min_dist4 is None or dist4 < min_dist4 + EPSILON:
                # We keep all objects whose distance is between the minimum and minimum+EPSILON
                if min_dist4 is None:
                    assert len(relevant_obj_ix4) == 0
                    min_dist4 = dist4
                    relevant_obj_ix4[ix] = dist4
                elif dist4 < min_dist4:
                    min_dist4 = dist4
                    relevant_obj_ix4 = {ti: td for ti, td in relevant_obj_ix4.items() if td <= min_dist4 + EPSILON}
                    relevant_obj_ix4[ix] = dist4
                else:
                    # min_dist4 <= dist4 < min_dist4 + EPSILON
                    relevant_obj_ix4[ix] = dist4

        if len(relevant_obj_ix4) == 0:
            self.errors['no_min_obj4'] += 1
            self.draw_error(picx, picy)
            return

        if utils.DEBUG:
            print(f"(main) Closest objects in 4d: {relevant_obj_ix4} dist={min_dist4}")

        if len(relevant_obj_ix4) == 1:
            min_obj_ix4 = list(relevant_obj_ix4)[0]
        else:
            self.errors['multiple_min_obj4'] += 1
            # This means there are multiple overlapping objects in the 4D space - get a random one
            min_obj_ix4 = random.choice(list(relevant_obj_ix4))
            if utils.XCHECK:
                tmp = random.choice(list(relevant_obj_ix4))
                if abs(relevant_obj_ix4[min_obj_ix4] - relevant_obj_ix4[tmp]) > EPSILON:
                    raise XCheckError("selected 4d objects not at same distance")

        min_obj4 = self.objects[min_obj_ix4]
        intersect_point_4d = ray_4d.get_line_point(min_dist4)
        if utils.XCHECK:
            if not min_obj4.body.includes_sub(intersect_point_4d, permission_level=1):
                raise XCheckError("min_obj4 does not contain intersect_point_4d")
            tmp_point = ray_4d.get_line_point(min_dist4 - .1)
            if not min_obj4.body.includes_sub(tmp_point, permission_level=0):
                raise XCheckError("min_obj4 contains a point closer than intersect_point_4d")

        self.img_arr[picy, picx, :] = min_obj4.get_color(point=intersect_point_4d, lights=self.lights, eye=self.cameras[1].focal)

    def save_img(self, filename: str = 'image.png'):
        # Save the image
        # img_max = np.max(img_arr, axis=(0,1)) + 1e-7 # per channel
        img_max = np.max(self.img_arr) + 1e-7 # overall max
        img_data = (self.img_arr / img_max * 255.).astype('B')
        img = Image.fromarray(img_data, mode="RGB") # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        img.save(filename)


    def coord2pix(self, c):
        """Convert a coordinate to a pixel index"""
        return int((c + self.img_range) / self.img_step + .5)

    def plot(self, x: float, y: float, c):
        """Set the color in the image at the given coordinates"""
        self.img_arr[self.coord2pix(y), self.coord2pix(x), :] = c

    def bigdot(self, y: int, x: int, c=(1,0,0)):
        """Draw a bigger dot on the image at the given coordinates"""
        self.img_arr[y,x,:] = c
        self.img_arr[y + 1, x, :] = c
        self.img_arr[y - 1, x, :] = c
        self.img_arr[y, x + 1, :] = c
        self.img_arr[y,x-1,:] = c

    def draw_error(self, picx: int, picy: int):
        """Mark the image for areas where errors occurred"""
        self.img_arr[picy, picx, :] = [1.1,0,0] if ((picx + picy) % 2) == 0 else [1.1,1.1,0]
