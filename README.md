
# Hypergeometry

A Python library to do geometric manipulations of arbitrarily-many-dimensional points and vectors,
and CGI tools to render a 4D scene as a 2D image.

## Quickstart


## Classes and object types

### Point

`Point(list[float])` - A point or vector

### Poly

`Poly(list[list[float]])` - A collection of points or vectors

### Span

`Span(org:Point, basis:Poly)` - A `basis` or frame of vectors anchored at a specific point (`org`) in space.
Can represent a subspace, or (via subclasses) a parallelotope or a simplex.

### Body(Span)

`Body(org:Point, basis:Poly)` - A subclass of Span. 
Contains operations common to simplices and parallelotopes.

### Simplex(Body)

`Simplex(org:Point, basis:Poly)` - A subclass of Body.
An n-dimensional simplex (generalization of the triangle and the tetrahedron) 
which is the collection of points around `org` along the vectors in `basis` with positive
coefficients whose sum it at most 1.

### Parallelotope(Body)

`Parallelotope(org:Point, basis:Poly)` - A subclass of Body.
An n-dimensional parallelotope (generalization of the parallelogram and the parallelepiped)
which is the collection of points around `org` along the vectors in `basis` with coefficients between 0 and 1.

### ObjectFace

```python
ObjectFace(
    body: Simplex,
    normal: Point,
    color: (R,G,B),
    surface: str
)
```
ObjectFace objects are the building blocks of our scenery.
We work with triangular (simplex) faces that are one dimension lower than the dimension of the scene,
so in a 4D scene, we work with 3D faces covering the surfaces of the 4D objects. 
The faces also need to be oriented, so we store their normal vector, as well as their
color, and the type of their surface that determines the calculation used for lighting effects
(e.g. matte, translucent, etc.)

## More info

Please read detailed explanations and background info in my blog
in posts tagged with Hypergeometry: https://onkeypress.blogspot.com/search/label/Hypergeometry .


