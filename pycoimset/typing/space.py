# PyCoimset: Python library for COntinuous IMprovement of SETs
#
# Copyright 2023 Mirko Hahn
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
Static typing protocols for similarity space definition.
'''

from types import NotImplementedType
from typing import Literal, Optional, Protocol, Self, TypeVar, final


from ..util.cache import cached_method


__all__ = [
    'SignedMeasure',
    'SimilarityClass',
    'SimilaritySpace',
]


Spc = TypeVar('Spc', bound='SimilaritySpace')


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

    @property
    def l1_norm(self) -> float:
        '''
        Calculate total variation norm.

        Notes
        -----
            This property is generally not cached. You should use `norm`
            instead.
        '''
        return self(self > 0) - self(self < 0)

    @property
    def linfty_norm(self) -> float:
        '''
        Calculate essential supremum of absolute density.

        Notes
        -----
            This property is generally not cached. You should use `norm`
            instead.
        '''
        raise NotImplementedError('Linfty norm has not been implemented.')

    @final
    @cached_method('_cache_norm')
    def norm(self, kind: Literal['L1', 'Linfty']) -> float:
        '''
        Calculate an Lq norm of the density function.

        This is a mixin method. You cannot override it. It uses caching with
        no lifetime limit. Therefore, it requires the signed measure to be
        immutable.

        Argument
        --------
        kind : 'L1' or 'Linfty'
            Indicates which norm to use. Only 'L1' is guaranteed to be
            implemented.

        Returns
        -------
            Value of the norm.

        Raises
        ------
        NotImplementedError
            Indicates that the norm has not been implemented for the signed
            measure type.
        '''
        if kind == 'Linfty':
            return self.linfty_norm
        else:
            return self.l1_norm

    def __call__(self, arg: SimilarityClass[Spc]) -> float:
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

    def __neg__(self) -> 'SignedMeasure[Spc]':
        '''Negative of measure.'''
        return self.__mul__(-1)


class SimilaritySpace(Protocol):
    '''
    Protocol for the underlying metric space of a :class:`SimilarityClass`.

    This may also encode a product similarity space.
    '''
    @property
    def measure(self) -> float:
        '''Measure of the universal class.'''
        ...

    @property
    def empty_class(self) -> SimilarityClass[Self]:
        '''Empty similarity class.'''
        ...

    @property
    def universal_class(self) -> SimilarityClass[Self]:
        '''Universal similarity class.'''
        ...
