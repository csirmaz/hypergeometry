
# Hypergeometry

A Python library to do geometric manipulations of arbitrarily-many-dimensional points and vectors,
and CGI tools to render a 4D scene as a 2D image.

## Quickstart

`main.py` contains the definition of a simple 4D scene and code to render it
into a 2D image.

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

### Camera

`Camera(space:Span, focd:float)` - Defines a camera.
A camera is defined by a Span spanning the whole space with an orthonormal basis.
The first D-1 vectors form the image pane, while the focal point is
`focd` (focal distance) away along the last vector.

### Light

`Light(p:Point)` - Defines a point light.

### Renderer

```python
Renderer(
    objects: list[ObjectFace],
    cameras: list[Camera],  # one for each dimension reduction
    lights: list[Light],
    img_range: float,  # the maximum 2D coordinate in the 2D image
    img_step: float  # the resolution of the 2D image
)
```
Groups objects, cameras and lights and contains logic to render a 4D scene
into a 2D image.

## More info

Please read detailed explanations and background info in the wiki at
https://github.com/csirmaz/hypergeometry/wiki


