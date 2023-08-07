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
    'ShiftedFunctional',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


class TransformedFunctional(Functional[Spc], Generic[Spc]):
    '''
    Applies an affine transformation to a functional.
    '''
    _func: Functional[Spc]
    _shift: float
    _scale: float

    def __init__(self, func: Functional[Spc], shift: float = 0.0,
                 scale: float = 1.0):
        self._func = func
        self._shift = shift
        self._scale = scale

    @property
    def base(self) -> Functional[Spc]:
        '''Underlying set functional.'''
        return self._func

    @property
    def shift(self) -> float:
        '''Shifting constant.'''
        return self._shift

    @shift.setter
    def shift(self, shift: float) -> None:
        self._shift = shift

    @property
    def scale(self) -> float:
        '''Scaling constant.'''
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

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

    @property
    def grad_tol_type(self) -> ErrorNorm:
        '''Gradient error norm.'''
        return self._func.grad_tol_type

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
