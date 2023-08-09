
# Initial unit tests

import numpy as np

from hypergeometry import Point, Poly, Span, Combination, Parallelotope, Simplex, loop_bin, select_of, loop_natural_bin, loop_many_to

def close(a, b):
    return np.allclose(np.array([a]), np.array([b]))

def point_test():
    p1 = Point([0, 0, 1])
    assert p1.scale(2).scale(.5).eq(p1)
    assert p1.dim() == 3
    assert p1.add(p1).scale(.5).eq(p1)
    assert p1.length() == 1
    assert p1.rotate((1,2), .5).allclose(Point([0, 1, 0]))
    p2 = p1.clone()
    p2.c[0] = 1
    assert not p1.eq(p2)
    assert not p1.allclose(p2)
    assert p2.ge(p1)
    assert p2.persp_reduce(-5).dim() == 2
    
    target=[
        Point([0,0]), Point([0,.5]), Point([0,1]),
        Point([.5,0]), Point([.5,.5]), Point([.5,1]),
        Point([1,0]), Point([1,.5]), Point([1,1]),
    ]
    for i, p in enumerate(Point.generate_grid(dim=2, steps=2)):
        assert target[i].eq(p)


def poly_test():
    p1 = Point([0,0,1])
    p2 = Point([0,1,0])
    p = Poly([p1, p2])
    assert p.dim() == 3
    assert p.sum().eq(p1.add(p2))
    assert p1.add(p2).scale(.5).eq(p.mean())
    assert p.is_orthonormal()
    assert not Poly([[3,4],[6,8]]).norm().is_orthonormal()
    assert Poly([p1.scale(2), p1.add(p2)]).make_basis().eq(p)
    assert p.apply_to(Point([1,2])).eq(Point([0,2,1]))
    assert p.apply_to(Poly([[1,2],[3,4]])).eq(Poly([[0,2,1],[0,4,3]]))
    assert p.persp_reduce(-10).allclose(Poly(p.p[:,:-1]))

    base2 = Poly.from_identity(3).rotate((0,1),.2).rotate((1,2),.3)
    assert base2.is_orthonormal()
    points = Poly([[50,60,70],[-1,-3,-2]])
    assert base2.apply_to(base2.extract_from(points)).allclose(points)
    assert base2.apply_to(base2.extract_from(points.at(0))).allclose(points.at(0))
    assert Poly([[1,0],[1,1]]).extract_from(Point([10.9, 31.4])).allclose(Point([-20.5, 31.4]))
    assert Poly([[1,0,0],[1,1,0]]).extract_from(Point([10.9, 31.4, 100])).allclose(Point([-20.5, 31.4]))
    assert Poly([[-1,0,0],[0,1,0]]).extract_from(Point([10.9, 31.4, 100])).allclose(Point([-10.9, 31.4]))
    
    assert p.is_independent()
    assert Poly.from_identity(5).is_independent()
    assert base2.is_independent()
    assert not Poly([p1, p1, p2]).is_independent()
    assert not Poly([
        Point([0,1,2]),
        Point([0,2,4])
    ]).is_independent()
    assert not Poly([
        Point([0,1,2]),
        Point([7,6,5]),
        Point([1,1,1])
    ]).is_independent()
    
    assert p.extend_to_square().is_independent(force=True)

def span_test():
    p = Point([1,1,1])
    v = Poly([[1,1,0], [0,0,1]])
    span = Span(org=p, basis=v)
    assert span.space_dim() == 3
    assert span.my_dim() == 2
    comb = Combination([[1,1,1], [2,2,1], [1,1,2]])
    assert span.as_combination().allclose(comb)
    assert Span.from_combination(comb).allclose(span)
    assert comb.space_dim() == 3
    assert comb.my_dim() == 2

    p2 = Point([3,3,2])
    p3 = Point([2,1,0])
    span2 = Span(org=p, basis=Poly([[1,1,0], [0,0,1], [0,1,0]]))
    assert span2.extract_from(p2).allclose(p3)
    assert span2.apply_to(p3).allclose(p2)

    p3 = Point([2,1])
    assert span.apply_to(p3).allclose(p2)


def util_test():
    
    assert close(1, 1)
    assert not close(1, 1.1)
    assert close(0, 0)
    
    def clonelist(iterator):
        return [list(x) for x in iterator]

    assert clonelist(loop_bin(1)) == [[0], [1]]
    assert clonelist(loop_bin(2)) == [[0,0], [0,1], [1,0], [1,1]]
    assert clonelist(select_of(2,4)) == [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    assert list(loop_natural_bin(2)) == [[0,0], [0,1], [1,1], [1,0]]
    assert list(loop_natural_bin(3)) == [[0,0,0], [0,0,1], [0,1,1], [0,1,0], [1,1,0], [1,1,1], [1,0,1], [1,0,0]]
    assert clonelist(loop_many_to(2, 3)) == [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]


def body_test():
    p = Parallelotope(org=Point([1,1]), basis=Poly([
        [2,0],
        [0,2]
    ]))
    assert p.distance_on_2d(Span.create_line([0,.1],[1,0])) is None
    assert p.distance_on_2d(Span.create_line([0,1],[1,0])) == 1
    assert p.distance_on_2d(Span.create_line([0,1.5],[1,0])) == 1
    assert p.distance_on_2d(Span.create_line([100,2],[-1,0])) == 97
    assert p.distance_on_2d(Span.create_line([1,4],[1,-1])) == 1
    assert p.distance_on_2d(Span.create_line([0,5],[1,-1])) == 2
    assert p.distance_on_2d(Span.create_line([1,4],[-1,-1])) is None
    
    assert p.includes(Point([1,1]))
    assert p.includes(Point([1.1,1]))
    assert p.includes(Point([3,3]))
    assert p.includes(Point([1.1,2.9]))
    assert not p.includes(Point([0,0]))
    assert not p.includes(Point([1,.9]))
    assert not p.includes(Point([1,3.1]))
    
    assert p.intersect_line(Span.create_line([0,0],[1,1])) == 1
    assert p.intersect_line(Span.create_line([0,0],[1,0])) is None
    assert p.intersect_line(Span.create_line([1,0],[0,1])) == 1
    assert p.intersect_line(Span.create_line([1,10],[0,-1])) == 7
    p2 = Parallelotope(org=Point([1,1,1]), basis=Poly([[2,0,0],[0,2,0]]))
    assert p2.intersect_line(Span.create_line([0,0,1],[1,1,0])) == 1
    assert close(p2.intersect_line(Span.create_line([0,0,0],[1,1,1])), 1)
    
    s = Simplex(org=Point([1,1]), basis=Poly([
        [2,0],
        [0,2]
    ]))
    assert s.includes(Point([1,1]))
    assert s.includes(Point([1.1,1.1]))
    assert s.includes(Point([2,2]))
    assert s.includes(Point([1,3]))
    assert s.includes(Point([1.1,2.9]))
    assert not s.includes(Point([1.1,3]))
    assert not s.includes(Point([2.1,2.1]))

    assert s.intersect_line(Span.create_line([0,0],[1,1])) == 1
    assert s.intersect_line(Span.create_line([0,0],[1,0])) is None
    assert s.intersect_line(Span.create_line([1,0],[0,1])) == 1
    assert s.intersect_line(Span.create_line([1,10],[0,-1])) == 7
    assert s.intersect_line(Span.create_line([1,3.1],[1,-1])) is None
    assert s.intersect_line(Span.create_line([1,3],[1,-1])) == 0
    assert close(s.intersect_line(Span.create_line([2,2],[-1,-1])), 0)
    assert close(s.intersect_line(Span.create_line([2.1,2.1],[-1,-1])), .1)
    assert close(s.intersect_line(Span.create_line([2.6,1.6],[-1,-1])), .1)
    
    s2 = Simplex(org=Point([1,2]), basis=Poly([[1,1], [2,0]]))
    s3 = Simplex(org=Point([1,2]), basis=Poly([[2,0], [1,1]]))
    for ts in [s2, s3]:
        assert close(ts.intersect_line(Span.create_line([0,0],[1,1])), 2)
        assert close(ts.intersect_line(Span.create_line([1,0],[2,2])), 1)
        assert ts.intersect_line(Span.create_line([1.1,0],[2,2])) is None
        assert close(ts.intersect_line(Span.create_line([1,4],[1,-1])), 1)
        assert close(ts.intersect_line(Span.create_line([0,4],[1,-1])), 1.5)
        assert ts.intersect_line(Span.create_line([1.1,4],[1,-1])) is None
        assert close(ts.intersect_line(Span.create_line([3,3],[-1,-1])), .5)


def line_test():
    # Test line opertions
    line1 = Span.create_line([1,2], [.2,.3])
    assert line1.get_line_point(2).allclose(Point([1.4, 2.6]))
    line2 = Span.create_line([7,9], [.41,.27])
    r = line1.intersect_lines_2d(line2, test=True)
    assert line1.get_line_point(r[0]).allclose(line2.get_line_point(r[1]))
    
# Since some operations involve random values, we repeat the tests
for i in range(100):
    point_test()
    poly_test()
    span_test()
    util_test()
    body_test()
    line_test()
print("OK")
