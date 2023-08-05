from typing import Iterable
import numpy as np

from hypergeometry.span import Body

class Simplex(Body):
    """An n-dimensional simplex defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi and Sum(xi) <= 1
    """
    
    def decompose(self) -> Iterable['Simplex']:
        """Return the n-1-dimensional faces
        Note: we don't worry about the orientation"""
        o = [
            Simplex(
                org=self.org,
                basis=self.basis.except_for(i)
            ) for i in range(self.my_dim())
        ]
        o.append(Simplex(
            org=self.org.add(self.basis.at(0)),
            basis=Poly(self.basis.p[1:] - self.basis.p[0])
        ))
        return o
    
