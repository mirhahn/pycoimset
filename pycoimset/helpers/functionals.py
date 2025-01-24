# PyCoimset: Python library for COntinuous IMprovement of SETs
#
# Copyright 2024 Mirko Hahn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Helpers used to modify functionals.
'''

import math
from typing import Generic, Optional, Sequence, TypeVar

import numpy
from numpy.typing import NDArray, ArrayLike

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
    def base_functional(self) -> Functional[Spc]:
        '''Underlying functional.'''
        return self._func

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


class weighted_sum(Functional[Spc], Generic[Spc]):
    '''
    Weighted sum of several functionals.
    '''
    _spc: Spc
    _gtype: ErrorNorm
    _c: NDArray[numpy.floating]
    _f: Sequence[Functional[Spc]]
    _vwgt: NDArray[numpy.floating]
    _gwgt: NDArray[numpy.floating]
    _vtol: float
    _gtol: float

    def __init__(self, func: Sequence[Functional[Spc]], coef: ArrayLike = 1.0,
                 val_wgt: ArrayLike = 1.0, grad_wgt: ArrayLike = 1.0):
        if len(func) == 0:
            raise ValueError('must provide at least one component functional')
        self._spc = func[0].input_space
        self._gtype = func[0].grad_tol_type
        for f in func[1:]:
            if f.input_space is not self._spc:
                raise ValueError(
                    'all components must use the same input space'
                )
            if f.grad_tol_type is not self._gtype:
                raise ValueError(
                    'all components must have the same error control type'
                )
        self._f = func
        self._c = numpy.broadcast_to(numpy.asarray(coef, dtype=float),
                                     len(func))
        self._vwgt = numpy.broadcast_to(numpy.asarray(val_wgt, dtype=float),
                                        len(func))
        self._gwgt = numpy.broadcast_to(numpy.asarray(grad_wgt, dtype=float),
                                        len(func))
        self._vtol = math.inf
        self._gtol = math.inf

    def __len__(self) -> int:
        '''Number of components.'''
        return len(self._f)

    def __getitem__(self, idx: int) -> Functional[Spc]:
        '''Component functional.'''
        return self._f[idx]

    @property
    def input_space(self) -> Spc:
        '''Input space.'''
        return self._spc

    @property
    def arg(self) -> Optional[SimilarityClass[Spc]]:
        '''Current argument.'''
        return self._f[0].arg

    @arg.setter
    def arg(self, arg: Optional[SimilarityClass[Spc]]) -> None:
        for f in self._f:
            if f.arg is not arg:
                f.arg = arg

    @property
    def val_tol(self) -> float:
        '''Value tolerance.'''
        return self._vtol

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        self._vtol = tol
        if not numpy.isfinite(tol):
            tol_comp = numpy.full(len(self._f), numpy.inf)
        else:
            wgt = abs(self._c * self._vwgt)
            tol_comp = tol * (wgt / numpy.sum(wgt))
        for f, vt in zip(self._f, tol_comp):
            f.val_tol = vt

    @property
    def grad_tol(self) -> float:
        '''Gradient tolerance.'''
        return self._gtol

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        self._gtol = tol
        if not numpy.isfinite(tol):
            tol_comp = numpy.full(len(self._f), numpy.inf)
        else:
            wgt = abs(self._c * self._gwgt)
            tol_comp = tol * (wgt / numpy.sum(wgt))
        for f, gt in zip(self._f, tol_comp):
            f.grad_tol = gt

    @property
    def grad_tol_type(self) -> ErrorNorm:
        '''Type of error control.'''
        return self._gtype

    def get_value(self) -> tuple[float, float]:
        '''Return value-error pair.'''
        val = numpy.empty(len(self._f), dtype=float)
        err = numpy.empty(len(self._f), dtype=float)
        for i, f in enumerate(self._f):
            val[i], err[i] = f.get_value()
        return numpy.inner(self._c, val), numpy.inner(numpy.abs(self._c), err)

    def get_gradient(self) -> tuple[SignedMeasure[Spc], float]:
        '''Return gradient-error pair.'''
        grad = None
        err = numpy.empty(len(self._f), dtype=float)
        for i, (c, f) in enumerate(zip(self._c, self._f)):
            g, err[i] = f.get_gradient()
            if grad is None:
                grad = c * g
            else:
                grad = grad + c * g
        assert grad is not None
        return grad, numpy.inner(numpy.abs(self._c), err)
