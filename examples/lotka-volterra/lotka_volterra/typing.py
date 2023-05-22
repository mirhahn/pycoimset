'''
Typing helpers for objects that behave like SciPy OdeSolution.
'''

from typing import Protocol

from numpy.typing import ArrayLike, NDArray


class OdeSolutionLike(Protocol):
    '''
    Behaves like `scipy.integrate.OdeSolution` with `deriv` extension.
    '''

    #: Time instants between which local interpolants are defined.
    #: Must be strictly increasing.
    ts: NDArray

    @property
    def t_min(self) -> float:
        '''Start of interpolation time range.'''
        return self.ts[0]

    @property
    def t_max(self) -> float:
        '''End of interpolation time range.'''
        return self.ts[-1]

    def __call__(self, t: ArrayLike) -> NDArray:
        '''Evaluate the solution.'''
        ...

    def deriv(self, t: ArrayLike) -> NDArray:
        '''Evaluate the first derivative of the solution.'''
        ...
