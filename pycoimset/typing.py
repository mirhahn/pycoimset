'''
Static type checking for user-defined types.
'''

import dataclasses

from enum import StrEnum
from types import NotImplementedType
from typing import cast, Callable, Optional, Protocol, Self, TypeVar


__all__ = [
    'ErrorNorm',
    'JSONSerializable',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'Functional',
]


Spc = TypeVar('Spc', bound='SimilaritySpace')


class JSONSerializable(Protocol):
    def toJSON(self) -> dict | list:
        '''Serialize the object into a JSON-compatible object.'''
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        raise NotImplementedError()

    @classmethod
    def fromJSON(cls, obj: dict | list) -> Self:
        '''Deserialize an object from JSON data.'''
        if isinstance(obj, list):
            return cast(Self, cls(*obj))
        return cast(Self, cls(**obj))


class SimilarityClass(Protocol[Spc]):
    '''
    Protocol class for the values of a vector of set-valued variables.

    Although named `SimilarityClass`, this protocol is expressly meant
    to support both single similarity classes and product similarity
    classes. Although a general implementation of product similarity
    classes is both conceivable and may be provided in the future. It
    is always better to provide an implementation that is tailored to
    the problem at hand.
    '''
    @property
    def space(self) -> Spc:
        '''Underlying measure space.'''
        ...

    @property
    def measure(self) -> float:
        '''Measure of the class.'''
        ...

    def subset(self, meas_low: float, meas_high: float,
               hint: Optional['SignedMeasure[Spc]'] = None
               ) -> 'SimilarityClass[Spc]':
        '''Choose subset within a given size range.'''
        ...

    def __invert__(self) -> 'SimilarityClass[Spc]':
        '''Return complement of this class.'''
        ...

    def __or__(self, other: 'SimilarityClass[Spc]'
               ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return union with another class.'''
        ...

    def __ror__(self, other: 'SimilarityClass[Spc]'
                ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return union with another class.'''
        return self.__or__(other)

    def __and__(self, other: 'SimilarityClass[Spc]'
                ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return intersection with another class.'''
        ...

    def __rand__(self, other: 'SimilarityClass[Spc]'
                 ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return intersection with another class.'''
        return self.__and__(other)

    def __sub__(self, other: 'SimilarityClass[Spc]'
                ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return difference with another class.'''
        ...

    def __rsub__(self, other: 'SimilarityClass[Spc]'
                 ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return difference with another class.'''
        ...

    def __xor__(self, other: 'SimilarityClass[Spc]'
                ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return symmetric difference with another class.'''
        ...

    def __rxor__(self, other: 'SimilarityClass[Spc]'
                 ) -> 'SimilarityClass[Spc] | NotImplementedType':
        '''Return symmetric difference with another class.'''
        return self.__xor__(other)


class SignedMeasure(Protocol[Spc]):
    '''
    Protocol for gradient measures.

    Like :class:`SimilarityClass`, this may be used to represent a
    vector of signed measures.
    '''
    @property
    def space(self) -> Spc:
        '''Underlying measure space.'''
        ...

    def __call__(self, arg: SimilarityClass[Spc]
                 ) -> float | NotImplementedType:
        '''Measure a given set.'''
        ...

    def __lt__(self, level: float
               ) -> SimilarityClass[Spc] | NotImplementedType:
        '''Return strict sublevel similarity class.'''
        ...

    def __le__(self, level: float
               ) -> SimilarityClass[Spc] | NotImplementedType:
        '''Return non-strict sublevel similarity class.'''
        ...

    def __gt__(self, level: float
               ) -> SimilarityClass[Spc] | NotImplementedType:
        '''Return strict superlevel similarity class.'''
        ...

    def __ge__(self, level: float
               ) -> SimilarityClass[Spc] | NotImplementedType:
        '''Return non-strict superlevel similarity class.'''
        ...

    def __add__(self, other: 'SignedMeasure[Spc]'
                ) -> 'SignedMeasure[Spc] | NotImplementedType':
        '''Add up with another signed measure.'''
        ...

    def __radd__(self, other: 'SignedMeasure[Spc]'
                 ) -> 'SignedMeasure[Spc] | NotImplementedType':
        '''Add up with another signed measure.'''
        return self.__add__(other)

    def __sub__(self, other: 'SignedMeasure[Spc]'
                ) -> 'SignedMeasure[Spc] | NotImplementedType':
        '''Subtract another signed measure.'''
        ...

    def __rsub__(self, other: 'SignedMeasure[Spc]'
                 ) -> 'SignedMeasure[Spc] | NotImplementedType':
        '''Subtract another signed measure.'''
        ...

    def __mul__(self, factor: float) -> 'SignedMeasure[Spc]':
        '''Scale with a constant factor.'''
        ...

    def __rmul__(self, factor: float) -> 'SignedMeasure[Spc]':
        '''Scale with a given scalar.'''
        return self.__mul__(factor)

    def __truediv__(self, divisor: float) -> 'SignedMeasure[Spc]':
        '''Scale with reciprocal of a scalar.'''
        ...


class SimilaritySpace(Protocol[Spc]):
    '''
    Protocol for the underlying metric space of a :class:`SimilarityClass`.

    This may also encode a product similarity space.
    '''
    @property
    def measure(self) -> float:
        '''Measure of the universal class.'''
        ...

    @property
    def empty_class(self) -> SimilarityClass[Spc]:
        '''Empty similarity class.'''
        ...

    @property
    def universal_class(self) -> SimilarityClass[Spc]:
        '''Universal similarity class.'''
        ...


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
        Estimate errr for a similarity class of given size.

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


class UnconstrainedStepFinder(Protocol[Spc]):
    '''Protocol for unconstrained step finders.'''

    @property
    def quality(self) -> float:
        '''Return quality constant of this step finder.'''
        ...

    @property
    def gradient(self) -> Optional[SignedMeasure[Spc]]:
        '''Gradient measure.'''
        ...

    @gradient.setter
    def gradient(self, grad: Optional[SignedMeasure[Spc]]) -> None:
        '''Set gradient measure and discard all cached results.'''
        ...

    @property
    def radius(self) -> float:
        '''Trust-region radius.'''
        ...

    @radius.setter
    def radius(self, r: float) -> None:
        '''Set trust-region radius and discard cache.'''
        ...

    @property
    def tolerance(self) -> float:
        '''Step finding tolerance.'''
        ...

    @tolerance.setter
    def tolerance(self, tol: float) -> None:
        '''Set step finding tolerance and discard cached values.'''
        ...

    def get_step(self) -> tuple[SimilarityClass[Spc], float]:
        '''
        Calculate and return step.

        Implementations should use cached results wherever possible,
        but must ensure that such results satisfy the current error
        tolerance.

        :return: A tuple of step and step finding error.
        :raise ValueError: `gradient` is not set.
        '''
        ...
