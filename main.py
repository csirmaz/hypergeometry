
import random

import hypergeometry.utils as utils
from hypergeometry import Point, Poly, Span, Parallelotope, Camera, ObjectFace, Light, Renderer
from hypergeometry.generators import create_prism, scatter_sphere, create_pyramid, create_box


def make_tree(x, y, z):
    trunk_color = (.6, .5, 0)
    leaf_color = (.4, 1., .3)
    zoom = .5
    tree_offset = 0
    trunk_radius = .15 * zoom
    trunk_height = 4 * zoom + .5*random.random()
    branch_height = 2.2 * zoom + tree_offset + .5*random.random()
    branch_horiz = 1.2 * zoom
    branch_vert = .8 * zoom
    branch_radius = .08 * zoom
    leaf_size = .15 * zoom
    mid_foliage_radius = 1.3 * zoom
    mid_foliage_num = 250
    side_foliage_radius = .8 * zoom
    side_foliage_num = 120

    pr = create_prism

    objs4 = [
        [pr(org=[x, y, z, tree_offset], i=3, r=trunk_radius, length=trunk_height, name='trunk'), trunk_color],
    ]

    if True:  # branches
        objs4.extend([
            [pr(org=[x, y, z, branch_height], i=0, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=1, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=2, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=3, r=branch_radius, length=branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=0, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=1, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=2, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],
            [pr(org=[x, y, z, branch_height], i=3, r=branch_radius, length=-branch_horiz, name='branch1'), trunk_color],

            [pr(org=[x+branch_horiz, y, z, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
            [pr(org=[x, y+branch_horiz, z, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
            [pr(org=[x, y, z+branch_horiz, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
            [pr(org=[x-branch_horiz, y, z, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
            [pr(org=[x, y-branch_horiz, z, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
            [pr(org=[x, y, z-branch_horiz, branch_height], i=3, r=branch_radius, length=branch_vert, name='branch2'), trunk_color],
        ])

    objs = []
    for box in objs4:
        objs.extend(ObjectFace.from_body(box[0], color=box[1]))

    if True:  # foliage
        objs.extend(scatter_sphere(org=[x, y, z, tree_offset + trunk_height], rad=mid_foliage_radius, n=mid_foliage_num, size=leaf_size, color=leaf_color))

        objs.extend(scatter_sphere(org=[x+branch_horiz, y, z, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[x, y+branch_horiz, z, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[x, y, z+branch_horiz, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))

        objs.extend(scatter_sphere(org=[x-branch_horiz, y, z, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[x, y-branch_horiz, z, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[x, y, z-branch_horiz, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))

    # ROTATE
    around = Point([x, y, z, 0])
    objs = [o.rotate(coords=[0, 1], rad=random.random()*.5, around=around) for o in objs]
    objs = [o.rotate(coords=[0, 2], rad=random.random()*.5, around=around) for o in objs]
    return objs


def make_scene():
    objs = []
    # ground
    objs.extend(ObjectFace.from_body(body=create_box([-3,2,-3,0], [6, 10, 6, -.1]), color=(.2, .8, 0) )) #, div=[1, 1, 1, 1]))
    # trees
    for x in range(2):
        for y in range(2):
            for z in range(2):
                    objs.extend(make_tree(x*5-2.5, y*5+5, z*5-2.5))
    return objs


def main():

    camera_height = .75

    renderer = Renderer(
        objects=make_scene(),
        cameras=[

            # x -> x' -> x" (horizontal)
            # y -> y' -> normal, 3>2D camera focal point
            # z -> normal, 4>3D camera focal point
            # w -> z' -> y" (vertical)

            # 3D -> 2D
            # z' of (x'y'z') maps to y" of (x"y") (vertical orientation)
            Camera(space=Span(org=Point([0, 0, camera_height]),
                              basis=Poly([[1, 0, 0], # x' to x"
                                          [0, 0, 1], # z' to y" (vertical)
                                          [0, 1, 0]  # y' normal - to focal point
                                          ])), focd=-5),
            # 4D -> 3D
            # w of (xyzw) maps to z' of (x'y'z') (vertical orientation)
            Camera(space=Span(org=Point([0, 0, 0, camera_height]),
                              basis=Poly([[1, 0, 0, 0], # x to x'
                                          [0, 1, 0, 0], # y to y'
                                          [0, 0, 0, 1], # w to z'
                                          [0, 0, 1, 0]  # z normal - to focal point
                                          ])), focd=-5),
        ],
        lights=[
            Light(Point([0,0,-3,4]))
        ],
        img_range=2.,
        img_step=.005,  # resolution
        dump_objects=False
    )

    # Choose between a quick wireframe image or raytracing
    if True:
        renderer.raytracing()
    else:
        renderer.draw_wireframe()

    # for picy ends
    renderer.save_img()
    print(f"Errors encountered: {renderer.errors}")


main()
utils.print_profile()
