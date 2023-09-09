
from PIL import Image
import numpy as np

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON, XCheckError
from hypergeometry import Point, Poly, Span, Parallelotope, Simplex, Camera, ObjectFace, Light


class Renderer:

    def __init__(self,
        objects: list[ObjectFace],
        cameras: list[Camera],
        lights: list[Light],
        img_range: float,
        img_step: float,
        dump_objects: bool = False,
    ):
        self.objects = objects
        self.cameras = cameras  # First 3D->2D, then 4D->3D, etc.
        self.lights = lights
        self.img_range = img_range # 2D coords [-a..a] x [-a..a]
        self.img_step = img_step
        self.errors = {'ray3d_intersect': 0, 'no_min_obj3': 0, 'ray4d_intersect': 0, 'no_min_obj4': 0}
        self.objects_proj = None  # dict[<dimension>: list[Span]]
        self.objects_dist = None  # dict[<object_ix>: float]

        self.image_size = int(self.img_range / self.img_step * 2.)
        print(f"Image size: {self.image_size}")
        self.img_arr = np.zeros((self.image_size, self.image_size, 3))

        self.prepare_projections(dump_objects=dump_objects)
        self.prepare_distances()

    def prepare_projections(self, dump_objects: bool = False):
        """Prepare the 3D and 2D projections of the objects"""
        self.objects_proj = {x: [] for x in range(self.objects[0].body.space_dim())}
        for ix, obj in enumerate(self.objects):
            bdy = obj.body
            dim = bdy.space_dim()
            if dump_objects:
                print(f"Object #{ix} dim={dim}: [{obj.body.genesis()}] {obj}")
            for camera in self.cameras[::-1]:
                bdy = camera.project(bdy)
                dim -= 1
                self.objects_proj[dim].append(bdy)
                if dump_objects:
                    print(f"  Projection of #{ix} to dim={dim}: {bdy}")

    def prepare_distances(self):
        """Get the distances of objects to the last (highest dimensional) camera's focal point"""
        self.objects_dist = {}
        focal = self.cameras[-1].focal
        for ix, obj in enumerate(self.objects):
            midp = obj.body.midpoint()
            self.objects_dist[ix] = midp.sub(focal).length()

    def get_relevant_2d(self, im_point_2d: Point) -> list[int]:
        """Get a list of object indices relevant for a given 2D point"""
        objs = []
        for ix, obj in enumerate(self.objects_proj[2]):
            if obj.includes_sub(im_point_2d, permission_level=0):
                objs.append(ix)
        return objs

    def process_img_pixel(self, picy: int, picx: int):
        xcheck_edge_diff = .05
        im_point_2d = Point([  # image point on 2D canvas
            picx * self.img_step - self.img_range,
            (self.image_size - picy) * self.img_step - self.img_range  # flip Y
        ])

        local_debug = False  # Set to true to log information about computing

        # First check which objects are relevant based on their 2D projection
        relevant_obj_ix2 = self.get_relevant_2d(im_point_2d)
        if len(relevant_obj_ix2) == 0:
            return

        # Use ray tracing
        ray_3d = self.cameras[0].ray(im_point_2d)

        if utils.DEBUG or local_debug:
            print(f"\n(main) 2d image point: picy={picy} picx={picx} {im_point_2d} Relevant objects in 2d: {relevant_obj_ix2}")
            print(f"(main) 3d ray: {ray_3d}")
            for ix in relevant_obj_ix2:
                print(f"    #{ix} = {self.objects_proj[2][ix].genesis()}")

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
                print(f"\n>>> ray_3d intersect inconsistency at picy={picy} picx={picx}, err={intersect_err3}")
                continue

            if utils.DEBUG or local_debug:
                print(f"(main) Object #{ix} intersects the 3d ray at dist={dist3} point={ray_3d.get_line_point(dist3)}")
            if utils.XCHECK:
                tmp_point = ray_3d.get_line_point(dist3)
                if not obj3.includes_sub(tmp_point, permission_level=1):
                    raise XCheckError("obj3 does not contain im_point_3d")
                tmp_point = ray_3d.get_line_point(dist3 - xcheck_edge_diff)
                if obj3.includes_sub(tmp_point, permission_level=0):
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

        if utils.DEBUG or local_debug:
            print(f"(main) Closest objects on the 3d ray: {relevant_obj_ix3} Min dist={min_dist3} 3d image point: {im_point_3d}")
            print(f"(main) 4d ray: {ray_4d}")
            for ix in relevant_obj_ix3:
                print(f"    #{ix} = {self.objects_proj[3][ix].genesis()}")

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
                print(f"\n>>> ray_4d intersect inconsistency at picy={picy} picx={picx}, err={intersect_err4}")
                continue

            if utils.DEBUG or local_debug:
                print(f"(main) Object #{ix} intersects the 4d ray at dist={dist4} point={ray_4d.get_line_point(dist4)}")
            if utils.XCHECK:
                tmp_point = ray_4d.get_line_point(dist4)
                if not obj4.body.includes_sub(tmp_point, permission_level=1):
                    raise XCheckError("obj4 does not contain point_4d")
                tmp_point = ray_4d.get_line_point(dist4 - xcheck_edge_diff)
                if obj4.body.includes_sub(tmp_point, permission_level=0):
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

        if utils.DEBUG or local_debug:
            print(f"(main) Closest objects in 4d: {relevant_obj_ix4} dist={min_dist4}")
            for ix in relevant_obj_ix4:
                print(f"    #{ix} = {self.objects[ix].body.genesis()} focal dist={self.objects_dist[ix]}")

        if len(relevant_obj_ix4) == 1:
            min_obj_ix4 = list(relevant_obj_ix4)[0]
        else:
            # This means there are multiple overlapping objects in the 4D space - this can happen when the 3D surface objects
            # in 4D space share a 2D face, which they can do. We want the object that obscures the other one just next to the 2D face,
            # but we can't assess that at the 2D face. As a proxy we use the distance of the midpoint to the camera's focal point,
            # as we're dealing with simplices.
            min_focal_dist = None
            min_obj_ix4 = None
            for ix in relevant_obj_ix4.keys():
                if min_focal_dist is None or min_focal_dist > self.objects_dist[ix]:
                    min_focal_dist = self.objects_dist[ix]
                    min_obj_ix4 = ix

        if utils.DEBUG or local_debug:
            print(f"(main) Chosen object: #{min_obj_ix4} [{self.objects[min_obj_ix4].body.genesis()}]")

        min_obj4 = self.objects[min_obj_ix4]
        intersect_point_4d = ray_4d.get_line_point(min_dist4)
        if utils.XCHECK:
            if not min_obj4.body.includes_sub(intersect_point_4d, permission_level=1):
                raise XCheckError("min_obj4 does not contain intersect_point_4d")
            tmp_point = ray_4d.get_line_point(min_dist4 - xcheck_edge_diff)
            if min_obj4.body.includes_sub(tmp_point, permission_level=0):
                raise XCheckError("min_obj4 contains a point closer than intersect_point_4d")

        self.img_arr[picy, picx, :] = min_obj4.get_color(point=intersect_point_4d, lights=self.lights, eye=self.cameras[1].focal)

    def save_img(self, filename: str = 'image.png'):
        # Save the image
        # img_max = np.max(img_arr, axis=(0,1)) + 1e-7 # per channel
        img_max = np.max(self.img_arr) + 1e-7 # overall max
        if img_max < 1.: img_max = 1.
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
