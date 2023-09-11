
import hypergeometry.utils as utils
from hypergeometry import Point, Poly, Span, Parallelotope, Camera, ObjectFace, Light, Renderer
from hypergeometry.generators import create_prism, scatter_sphere, create_pyramid, create_box


def sample():
    #org = Point.zeros(4)
    #basis = Poly.from_identity(4)
    #body = Parallelotope(org=org, basis=basis)
    body = create_prism(org=[0,0,0,0], i=3, r=.1, length=.1)
    return list(ObjectFace.from_body(body=body, color=(1,1,0)))


def make_house():
    house_width = 2.
    wall_thickness = .1
    house_height = 2.
    roof_height = 3.
    house_offset = -1.5
    wall_color = (1., 1., 1.)
    roof_color = (.8, .1, 0)

    objs = []
    
    walls = []
    # Wall with door
    # A 4D wall has the size (width, width, thickness, height)
    # Windows and doors can be anywhere on its 3D surface, cutting through the thickness
    # Windows inside the 3D surface will not be visible on a 2D image
    # Here we construct full-height doors in the middle third
    h = house_width
    t = wall_thickness
    # Full walls
    # create_box([-h/2, -h/2, -h/2, house_offset], [h, h, t, house_height])
    # create_box([-h/2, -h/2, -h/2, house_offset], [h, t, h, house_height])
    # create_box([-h/2, -h/2, -h/2, house_offset], [t, h, h, house_height])
    # create_box([h/2, h/2, h/2, house_offset], [-h, -h, -t, house_height])
    # create_box([h/2, h/2, h/2, house_offset], [-h, -t, -h, house_height])
    # create_box([h/2, h/2, h/2, house_offset], [-t, -h, -h, house_height])

    for o in walls:
        objs.extend(ObjectFace.from_body(body=o, color=wall_color))

    # roof
    objs.extend(create_pyramid(
        base=create_box([-house_width/2, -house_width/2, -house_width/2, house_height + house_offset], [house_width, house_width, house_width, None]),
        apex=[0, 0, 0, roof_height + house_offset],
        color=roof_color
    ))

    # ROTATE
    objs = [o.rotate(coords=[0, 1], rad=.4) for o in objs]
    objs = [o.rotate(coords=[0, 2], rad=.3) for o in objs]
    objs = [o.rotate(coords=[2, 3], rad=-.1) for o in objs]
    return objs


def make_tree():
    trunk_color = (.6, .5, 0)
    leaf_color = (.4, 1., .3)
    tree_offset = -1.5
    trunk_radius = .2
    trunk_height = 2.5
    branch_height = 1.5 + tree_offset
    branch_horiz = 1.2
    branch_vert=.8
    branch_radius = .1
    leaf_size = .15
    mid_foliage_radius = 1.3
    mid_foliage_num = 150
    side_foliage_radius = .8
    side_foliage_num = 80

    pr = create_prism

    objs4 = [
        [pr(org=[0,0,0,tree_offset], i=3, r=trunk_radius, length=trunk_height, name='trunk'), trunk_color],

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

    if True:
        objs.extend(scatter_sphere(org=[0, 0, 0, tree_offset + trunk_height], rad=mid_foliage_radius, n=mid_foliage_num, size=leaf_size, color=leaf_color))

        objs.extend(scatter_sphere(org=[branch_horiz, 0, 0, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[0, branch_horiz, 0, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[0, 0, branch_horiz, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))

        objs.extend(scatter_sphere(org=[-branch_horiz, 0, 0, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[0, -branch_horiz, 0, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))
        objs.extend(scatter_sphere(org=[0, 0, -branch_horiz, branch_height + branch_vert], rad=side_foliage_radius, n=side_foliage_num, size=leaf_size, color=leaf_color))

    # ROTATE
    objs = [o.rotate(coords=[0, 1], rad=.4) for o in objs]
    objs = [o.rotate(coords=[0, 2], rad=.3) for o in objs]
    objs = [o.rotate(coords=[2, 3], rad=.1) for o in objs]
    return objs

def make_ground():
    objs = []
    objs.extend(ObjectFace.from_body(body=create_box([0,-11,0,0], [2, 20, -.1, -.1]), color=(.2, .8, 0)))
    return objs

def main():

    camera_height = 1.

    renderer = Renderer(
        objects=make_ground(), #make_tree(),
        cameras=[
            # 3D -> 2D
            # z' of (x'y'z') maps to y" of (x"y") (vertical orientation)
            Camera(space=Span(org=Point([0, 0, camera_height]),
                              basis=Poly([[1, 0, 0], # x' to x"
                                          [0, 0, 1], # z' to y" (vertical)
                                          [0, 1, 0]  # y' normal - to focal point
                                          ])), focd=-10),
            # 4D -> 3D
            # w of (xyzw) maps to z' of (x'y'z') (vertical orientation)
            Camera(space=Span(org=Point([0, 0, 0, camera_height]),
                              basis=Poly([[1, 0, 0, 0], # x to x' [to x"]
                                          [0, 1, 0, 0], # y to y' [to normal - to focal point]
                                          [0, 0, 0, 1], # w to z' [to y"] (vertical)
                                          [0, 0, 1, 0]  # z normal - to focal point
                                          ])), focd=-10),
        ],
        lights=[
            Light(Point([0,0,-3,4]))
        ],
        img_range=2.,
        img_step=.01,
        dump_objects=False
    )

    renderer.draw_wireframe()
    # renderer.raytracing()

    # for picy ends
    renderer.save_img()
    print(f"Errors encountered: {renderer.errors}")


#import cProfile
#cProfile.run('main()')
main()
utils.print_profile()
