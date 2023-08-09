
from hypergeometry import Point, Poly, Span, Combination, Parallelotope, Simplex

box = Parallelotope.create_box

# Define our world

objects = [
    box([0,0,0,0], [1,1,1,1])
]

