
import time

import hypergeometry.utils as utils
from hypergeometry.utils import EPSILON
from hypergeometry import Point, Poly, Span, Parallelotope, Simplex, Camera, ObjectFace, Light, Renderer

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

    renderer = Renderer(
        objects=list(ObjectFace.from_body(Parallelotope.create_box([1.,-.5,-1.,-1.],[1,1,2,2], name='BOX'), color=[1,1,1], use_face_colors=True)),
        cameras=[
            # 3D -> 2D
            # z of (xyz) maps to y' of (x'y') (vertical orientation)
            Camera(space=Span(org=Point([0, 0, 0]),
                              basis=Poly([[1, 0, 0], # to x
                                          [0, 0, 1], # to y
                                          [0, 1, 0]  # normal
                                          ])), focd=-10),
            # 4D -> 3D
            # w of (xyzw) maps to z' of (x'y'z') (vertical orientation)
            Camera(space=Span(org=Point([0, 0, 0, 0]),
                              basis=Poly([[1, 0, 0, 0], # to x'
                                          [0, 1, 0, 0], # to y'
                                          [0, 0, 0, 1], # to z'
                                          [0, 0, 1, 0]  # normal
                                          ])), focd=-10),
        ],
        lights=[
            Light(Point([0,-5,0,-5]))
        ],
        img_range=2.,
        img_step=.01,
        dump_objects=True
    )

    print("Processing pixels")
    start_time = time.time()
    percent_done = 0
    for picy in range(renderer.image_size): # pixel
        for picx in range(renderer.image_size): # pixel
            renderer.process_img_pixel(picy=picy, picx=picx)
        # for picx ends
        percent = int(picy / renderer.image_size * 100 + .5)
        if percent > percent_done:
            percent_done = percent
            spent_time = time.time() - start_time
            remaining_time = spent_time / percent_done * (100 - percent_done)
            print(f"{percent}% {remaining_time:.0f} s remaining, {renderer.errors} errors so far")

    # DEBUG
    renderer.bigdot(200, 298, [1,0,0])
    renderer.bigdot(200, 310, [1,0,0])

    # for picy ends
    renderer.save_img()
    print(f"Errors encountered: {renderer.errors}")


# import cProfile
# cProfile.run('main()')
main()
utils.print_profile()
