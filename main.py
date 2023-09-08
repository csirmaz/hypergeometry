
import time

import hypergeometry.utils as utils
from hypergeometry import Point, Poly, Span, Parallelotope, Camera, ObjectFace, Light, Renderer
from hypergeometry.generators import create_prism

def make_tree():
    trunk_color = (.6, .5, 0)
    leaves_color = (.4, 1., .3)
    trunk_radius = .2
    trunk_height = 2.5
    branch_height = 1.5
    branch_horiz = 1.2
    branch_vert=.8
    branch_radius = .1

    pr = create_prism

    objs4 = [
        [pr(org=[0,0,0,0], i=3, r=trunk_radius, length=trunk_height, name='trunk'), trunk_color],

        [pr(org=[0, 0, 0, branch_height], i=0, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=1, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=2, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=3, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=0, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=1, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=2, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
        [pr(org=[0, 0, 0, branch_height], i=3, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],

        [pr(org=[branch_horiz, 0, 0, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        [pr(org=[0, branch_horiz, 0, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        [pr(org=[0, 0, branch_horiz, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        [pr(org=[-branch_horiz, 0, 0, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        [pr(org=[0, -branch_horiz, 0, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        [pr(org=[0, 0, -branch_horiz, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
    ]

    objs = []
    for box in objs4:
        objs.extend(ObjectFace.from_body(box[0], color=box[1]))

    return objs


def main():

    renderer = Renderer(
        objects=make_tree(),
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
        dump_objects=False
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

    # for picy ends
    renderer.save_img()
    print(f"Errors encountered: {renderer.errors}")


# import cProfile
# cProfile.run('main()')
main()
utils.print_profile()
