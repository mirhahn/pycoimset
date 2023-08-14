'''
Helpers used to modify functionals.
'''

import math
from typing import Generic, Optional, TypeVar

from ..typing import (
    ErrorNorm,
    Functional,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
)


__all__ = [
    'transform',
    'with_safety_factor',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


class ProxyBase(Functional[Spc], Generic[Spc]):
    '''
    Base class for proxy functionals.
    '''
    _func: Functional[Spc]

    def __new__(cls, func: Functional[Spc]):
        obj = super().__new__(cls)
        obj._func = func
        return obj

    @property
    def input_space(self) -> Spc:
        '''Underlying similarity space.'''
        return self._func.input_space

    @property
    def arg(self) -> Optional[SimilarityClass[Spc]]:
        '''Current argument.'''
        return self._func.arg

    @arg.setter
    def arg(self, arg: Optional[SimilarityClass[Spc]]) -> None:
        self._func.arg = arg

    @property
    def val_tol(self) -> float:
        '''Value tolerance.'''
        return self._func.val_tol

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        '''Value tolerance.'''
        self._func.val_tol = tol

    @property
    def grad_tol(self) -> float:
        '''Gradient tolerance.'''
        return self._func.grad_tol

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        '''Gradient tolerance.'''
        self._func.grad_tol = tol

    @property
    def grad_tol_type(self) -> ErrorNorm:
        '''Gradient error norm.'''
        return self._func.grad_tol_type

    def get_value(self) -> tuple[float, float]:
        '''Get current functional value.'''
        return self._func.get_value()

    def get_gradient(self) -> tuple[SignedMeasure[Spc], float]:
        '''Get current functional value.'''
        return self._func.get_gradient()


class transform(ProxyBase[Spc], Generic[Spc]):
    '''
    Applies an affine transformation to a functional.
    '''
    _shift: float
    _scale: float

    def __new__(cls, func: Functional[Spc], shift: float = 0.0,
                scale: float = 1.0) -> Functional[Spc]:
        if shift == 0.0 and scale == 1.0:
            return func
        while isinstance(func, transform):
            shift += scale * func._shift
            scale *= func._scale
            func = func._func
        obj = super().__new__(cls, func)
        obj._shift = shift
        obj._scale = scale
        return obj

    @property
    def val_tol(self) -> float:
        '''Value tolerance.'''
        if self._scale == 0.0:
            return math.inf
        else:
            return self._func.val_tol * abs(self._scale)

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        '''Value tolerance.'''
        if self._scale == 0.0:
            self._func.val_tol = math.inf
        else:
            self._func.val_tol = tol / abs(self._scale)

    @property
    def grad_tol(self) -> float:
        '''Gradient tolerance.'''
        if self._scale == 0.0:
            return math.inf
        else:
            return self._func.grad_tol * abs(self._scale)

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        '''Gradient tolerance.'''
        if self._scale == 0.0:
            self._func.grad_tol = math.inf
        else:
            self._func.grad_tol = tol / abs(self._scale)

    def get_value(self) -> tuple[float, float]:
        '''Get current functional value.'''
        val, err = self._func.get_value()
        return self._scale * val + self._shift, abs(self._scale) * err

    def get_gradient(self) -> tuple[SignedMeasure[Spc], float]:
        '''Get current functional value.'''
        val, err = self._func.get_gradient()
        if self._scale == 1.0:
            return val, err
        else:
            return self._scale * val, abs(self._scale) * err


class with_safety_factor(ProxyBase[Spc], Generic[Spc]):
    '''
    Applies a safety factor to error estimates of a functional.

    Parameters
    ----------
    func : Functional[Spc]
        Base functional.
    factor : float
        Safety factor.
    grad_factor : float, optional
        Safety factor for gradient. Defaults to `factor`.
    '''
    _vfac: float
    _gfac: float

    def __new__(cls, func: Functional[Spc], factor: float,
                grad_factor: Optional[float] = None) -> Functional[Spc]:
        if grad_factor is None:
            grad_factor = factor
        if factor == 1.0 and grad_factor == 1.0:
            return func
        while isinstance(func, with_safety_factor):
            factor *= func._vfac
            grad_factor *= func._gfac
            func = func._func
        obj = super().__new__(cls, func)
        obj._vfac = factor
        obj._gfac = grad_factor
        return obj

    @property
    def val_tol(self) -> float:
        '''Tolerance for evaluation.'''
        return self._func.val_tol * self._vfac

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        self._func.val_tol = tol / self._vfac

    @property
    def grad_tol(self) -> float:
        '''Tolerance for gradient evaluation.'''
        return self._func.grad_tol * self._gfac

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        self._func.grad_tol = tol / self._gfac

    def get_value(self) -> tuple[float, float]:
        '''Get functional value and error.'''
        val, err = self._func.get_value()
        return val, err * self._vfac

    def get_gradient(self) -> tuple[SignedMeasure[Spc], float]:
        grad, err = self._func.get_gradient()
        return grad, err * self._gfac
