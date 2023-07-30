from typing import Iterable

from hypergeometry.span import Span

class Parallelotope(Span):
    """An n-dimensional parallelotope defined as n vectors from a
    given point (org):
    X = Org + x0*V0 + ... + xn*Vn
    where 0 <= xi <= 1
    """
    
    def decompose(self) -> Iterable['Parallelotope']:
        """Return the n-1-dimensional faces.
        We don't worry about the orientation.
        """
        inv = self.basis.scale(-1)
        org2 = self.org.add(self.basis.sum())
        return ([
            Parallelotope(
                org=self.org,
                basis=self.basis.except_for(i)
            ) for i in range(self.my_dim())
        ] + [
            Parallelotope(
                org=org2,
                basis=inv.except_for(i)
            ) for i in range(self.my_dim())
        ])
        

