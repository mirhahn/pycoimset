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
Static typing helpers for defining differentiable set functionals.
'''

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Callable, Generic, Optional, Protocol, TypeVar

from .space import SignedMeasure, SimilarityClass, SimilaritySpace


__all__ = [
    'Constraint',
    'Operator',
    'ErrorNorm',
    'Functional',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


class Operator(Enum):
    '''
    Comparison operator for constraints.
    '''
    is_satisfied: Callable[[float], bool]
    is_strictly_satisfied: Callable[[float], bool]
    violation: Callable[[float], float]
    
    def __init__(self,
                 loose_test: Callable[[float], bool],
                 strict_test: Callable[[float], bool],
                 violation: Callable[[float], float]):
        self.is_satisfied = loose_test
        self.is_strictly_satisfied = strict_test
        self.violation = violation

    LESS_THAN = (
        lambda g: g <= 0.0,
        lambda g: g < 0.0,
        lambda g: g
    )
    '''Indicates a less-than-or-equal constraint.'''
    EQUAL_TO = (
        lambda g: g == 0.0,
        lambda _: False,
        lambda g: abs(g)
    )
    '''Indicates an equality constraint.'''
    GREATER_THAN = (
        lambda g: g >= 0.0,
        lambda g: g > 0.0,
        lambda g: -g
    )
    '''Indicates a greater-than-or-equal constraint.'''


@dataclass(frozen=True)
class Constraint(Generic[Spc]):
    '''
    Description of a differentiable constraint.

    Parameters
    ----------
    func : Functional[Spc]
        Differentiable set functional.
    op : Operator
        Comparison operator.
    shift : float
        Shift constant.
    '''
    func: 'Functional[Spc]'
    op: Operator
    shift: float


class ErrorNorm(StrEnum):
    '''
    Enumeration of error norms for gradient error control.

    Gradient error can be controlled using several norms. Depending on
    which norm is chosen, different rules apply with regard to how
    tolerances and error estimates are calculated. This is encapsulated
    in this class.
    '''
    L1 = ('l1', lambda _, e: e, lambda _, e: e)
    Linfty = ('linfty', lambda mu, e: mu * e, lambda mu, e: e / mu)

    _err: Callable[[float, float], float]
    _tol: Callable[[float, float], float]

    def __new__(cls, value: str,
                error: Callable[[float, float], float],
                tolerance: Callable[[float, float], float]
                ):
        obj = str.__new__(cls)
        obj._value_ = value
        obj._err = error
        obj._tol = tolerance
        return obj

    def estimated_error(self, measure: float, error_norm: float) -> float:
        '''
        Estimate error for a similarity class of given size.

        :param measure: Measure of the similarity class.
        :type measure: float
        :param error_norm: Value of overall error norm.
        :type error_norm: float
        '''
        return self._err(measure, error_norm)

    def required_tolerance(self, measure: float, error_bound: float) -> float:
        '''
        Estimate error tolerance required to guarantee given error.

        :param measure: Measure for which the error bound should be
                        guaranteed.
        :type measure: float
        :param error_bound: Desired error bound.
        :type error_bound: float
        '''
        return self._tol(measure, error_bound)


class Functional(Protocol[Spc]):
    '''
    Protocol for a set functional.
    '''
    @property
    def input_space(self) -> Spc:
        '''Underlying similarity space of input.'''
        ...

    @property
    def arg(self) -> Optional[SimilarityClass[Spc]]:
        '''Current argument.'''
        ...

    @arg.setter
    def arg(self, arg: Optional[SimilarityClass[Spc]]) -> None:
        '''Set current argument and invalidate all cached results.'''
        ...

    @property
    def val_tol(self) -> float:
        '''Functional value evaluation tolerance.'''
        ...

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        '''Set functional value evaluation tolerance and invalidate cache.'''
        ...

    @property
    def grad_tol_type(self) -> ErrorNorm:
        '''Indicate type of gradient tolerance enforcement.'''
        ...

    @property
    def grad_tol(self) -> float:
        '''Functional gradient tolerance.'''
        ...

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        '''Set functional gradient tolerance and invalidate cache.'''
        ...

    def get_value(self) -> tuple[float, float]:
        '''
        Calculate and return value of functional at current argument.

        Implementations should use cached values where possible, but should
        ensure that cached values satisfy current error tolerances.

        :return: A tuple of objective value and error bound.
        :raise ValueError: `arg` has not been set.
        '''
        ...

    def get_gradient(self) -> tuple[SignedMeasure[Spc], float]:
        '''
        Calculate and return gradient of functional at current argument.

        Implementations should use cached values where possible, but
        should ensure that cached values satisfy current error
        tolerances.

        :return: A tuple of gradient and error bound.
        :raise ValueError: `arg` has not been set.
        '''
        ...

    def __le__(self, shift: float) -> Constraint[Spc]:
        '''
        Create less-than-or-equal constraint.
        '''
        return Constraint[Spc](self, Operator.LESS_THAN, shift)

    def __ge__(self, shift: float) -> Constraint[Spc]:
        '''
        Create greater-than-or-equal constraint.
        '''
        return Constraint[Spc](self, Operator.GREATER_THAN, shift)
