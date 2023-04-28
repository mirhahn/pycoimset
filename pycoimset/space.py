#
# PyCoimset - Python library for optimization with set-valued variables
# Copyright 2023 Mirko Hahn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
'''
Base types for measure spaces.
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
import operator
from typing import Callable, Optional

__all__ = [
    'SignedMeasure',
    'SimilarityClass',
    'SimilaritySpace',
]


class SimilarityClass(ABC):
    '''
    Abstract base class for similarity classes.

    Similarity classes are the residual classes of measurable sets with
    respect to the equivalence relation "equal up to a nullset." Their
    encoding and, by extension, their implementation depends strongly on the
    nature and discretization of the underlying measure space. Therefore,
    PyCoimset only interacts with similarity classes through a narrow
    interface prescribed by this class.

    Some of the conventional arithmetic operators are overriden with set
    operations to provide a shorthand for the member functions::

        a | b    # Union
        a & b    # Intersection
        a - b    # Set difference
        a ^ b    # Symmetric difference
        ~a       # Complement
    '''
    _space: 'SimilaritySpace'

    def __init__(self, space: 'SimilaritySpace'):
        super().__init__()
        self._space = space

    @property
    def space(self) -> 'SimilaritySpace':
        '''Containing similarity space.'''
        return self._space

    @abstractmethod
    def choose_subset(self, measure_low: float, measure_high: float,
                      hint: Optional['SignedMeasure'] = None
                      ) -> 'SimilarityClass':
        '''
        Return an arbitrary subset within a certain range of measures.

        This method is used to generate filler sets during step finding. The
        precise method of choosing the subset is irrelevant as long as a
        similarity class of appropriate measure is returned. Implementations
        must refine meshes as needed and may only throw an exception if doing
        so is no longer possible due to technical issues.

        :param measure_low: Lower bound of the returned class' measure.
        :type measure_low: float
        :param measure_high: Upper bound of the returned class' measure.
        :type measure_high: float
        :param hint: Signed measure that may be used to choose a "good" set.
            By convention, points with low density function value are
            "better."
        :type hint: :class:`SignedMeasure`, optional
        :return: A similarity class that is an essential subset of this one
            and whose measure is between `measure_low` and `measure_high`.
        :rtype: :class:`SimilarityClass`
        :raise ValueError: Indicates that either `measure_low` is not less
            than `measure_high` or that `measure_low` exceeds the measure of
            the set.
        :raise MemoryError: Indicates that necessary refinement is not
            possible due to memory limits.
        :raise RuntimeError: Indicates that necessary refinement is not
            possible due to machine precision limitations.
        '''
        pass

    @abstractmethod
    def measure(self) -> float:
        '''
        Return measure of the similarity class using the default measure.
        '''
        pass

    @abstractmethod
    def copy(self, deep: bool = False) -> 'SimilarityClass':
        '''
        Create a copy of the similarity class.

        :param deep: Make a deep instead of a shallow copy. Defaults to
            `False`.
        :type deep: bool, optional
        :return: A copy of this similarity class. If `deep` is `False`, then
            the copy may share resources with the original. Otherwise, it may
            not.
        :rtype: :class:`SimilarityClass`
        '''
        pass

    @abstractmethod
    def union(self, other: 'SimilarityClass') -> 'SimilarityClass':
        '''
        Return union with another similarity class.

        :param other: Other similarity class to unite with.
        :type other: :class:`SimilarityClass`
        :return: New similarity class object representing the union.
        :rtype: :class:`SimilarityClass`
        :raise NotImplementedError: Union is not implemented for the given
            types.
        '''
        pass

    @abstractmethod
    def intersection(self, other: 'SimilarityClass') -> 'SimilarityClass':
        '''
        Return intersection with another similarity class.

        :param other: Other similarity class to intersect with.
        :type other: :class:`SimilarityClass`
        :return: New similarity class object representing the intersection.
        :rtype: :class:`SimilarityClass`
        :raise NotImplementedError: Intersection is not implemented for the
            given types.
        '''
        pass

    @abstractmethod
    def difference(self, other: 'SimilarityClass') -> 'SimilarityClass':
        '''
        Return set difference with another similarity class.

        :param other: Similarity class to subtract.
        :type other: :class:`SimilarityClass`
        :return: New similarity class object representing the difference.
        :rtype: :class:`SimilarityClass`
        :raise NotImplementedError: Set difference is not implemented for the
            given types.
        '''
        pass

    @abstractmethod
    def symmdiff(self, other: 'SimilarityClass') -> 'SimilarityClass':
        '''
        Return symmetric difference with another similarity class.

        :param other: Similarity class to form symmetric difference with.
        :type other: :class:`SimilarityClass`
        :return: New similarity class object representing the symmetric
            difference.
        :rtype: :class:`SimilarityClass`
        :raise NotImplementedError: Symmetric difference is not implemented
            for the given types.
        '''
        pass

    @abstractmethod
    def complement(self) -> 'SimilarityClass':
        '''
        Return complement.

        :return: New similarity class object representing the complement.
        :rtype: :class:`SimilarityClass`
        '''
        pass

    def __copy__(self) -> 'SimilarityClass':
        return self.copy(False)

    def __deepcopy__(self) -> 'SimilarityClass':
        return self.copy(True)

    def __or__(self, other) -> 'SimilarityClass':
        if not isinstance(other, SimilarityClass):
            raise NotImplementedError()
        return self.union(other)

    def __and__(self, other) -> 'SimilarityClass':
        if not isinstance(other, SimilarityClass):
            raise NotImplementedError()
        return self.intersection(other)

    def __sub__(self, other) -> 'SimilarityClass':
        if not isinstance(other, SimilarityClass):
            raise NotImplementedError()
        return self.difference(other)

    def __xor__(self, other) -> 'SimilarityClass':
        if not isinstance(other, SimilarityClass):
            raise NotImplementedError()
        return self.symmdiff(other)

    def __invert__(self) -> 'SimilarityClass':
        return self.complement()


class SignedMeasure(ABC):
    '''
    Abstract base class for signed measures.

    Signed measures are the closest analogue of a linear functional in a
    similarity space. As such, they are used to represent gradients in
    optimization over such spaces.

    By convention, every signed measure encoded by this class is assumed to be
    absolutely continuous with respect to the underlying measure space's
    measure and therefore representable by a measurable density function. One
    of the primary functions of :class:`SignedMeasure` is to provide access to
    the density function's level sets.

    Some of the comparison operators have been overridden to yield level sets
    instead of booleans::

        a == b    # Exact level set
        a <= b    # Sublevel set
        a >= b    # Superlevel set
        a < b     # Strict sublevel set
        a > b     # Strict superlevel set

    These operations should be implemented if `b` is either of type `float` or
    :class:`SignedMeasure`.

    In addition, the class allows for rudimentary linear combination using the
    addition, subtraction, and multiplication operators, though these should
    be used with care because they are not evaluated lazily.
    '''

    @abstractmethod
    def measure(self, klass: SimilarityClass) -> float:
        '''
        Return measure of a similarity class.

        This is a well-defined value if and only if the signed measure is
        absolutely continuous with respect to the measure space's underlying
        measure.

        :param klass: Similarity class to be measured.
        :type klass: :class:`SimilarityClass`
        :return: The measure of the similarity class according to this signed
            measure.
        :rtype: float
        :raise NotImplementedError: The measure does not support this kind of
            similarity class.
        '''
        pass

    @abstractmethod
    def scale(self, alpha: float) -> 'SignedMeasure':
        '''
        Return a scaled version of this measure.

        :param alpha: Scaling coefficient.
        :type alpha: float
        :returns: New signed measure object representing a scaled version of
            this signed measure.
        :rtype: :class:`SignedMeasure`

        .. note::

            This may generate a full copy of the underlying data and is
            therefore very likely to be slow.
        '''
        pass

    @abstractmethod
    def add(self, other: 'SignedMeasure') -> 'SignedMeasure':
        '''
        Add with another signed measure.

        :param other: Other summand.
        :type other: :class:`SignedMeasure`
        :returns: New signed measure object representing the sum of both
            signed measures.
        :rtype: :class:`SignedMeasure`

        .. note::

            This may generate a full copy of the underlying data and is
            therefore very likely to be slow.
        '''
        pass

    @abstractmethod
    def sub(self, other: 'SignedMeasure') -> 'SignedMeasure':
        '''
        Return difference with another signed measure.

        :param other: The subtrahend.
        :type other: :class:`SignedMeasure`
        :returns: New signed measure object representing the difference of
            both signed measures.
        :rtype: :class:`SignedMeasure`

        .. note::

            This may generate a full copy of the underlying data and is
            therefore very likely to be slow.
        '''
        pass

    @abstractmethod
    def level_set(self, cmp: Callable[[float, float], bool],
                  other: 'float | SignedMeasure') -> SimilarityClass:
        '''
        Construct a level set.

        This is a single function to generate exact level sets as well as
        strict and non-strict super- and sublevel sets. The parameter `cmp` is
        used to determine which it is. `cmp` is a comparator that compares
        individual floats in the desired way.

        In order for this method to work properly, the return value of `cmp`
        must be constant along any path in its input space that does not cross
        the main diagonal, i.e., it may only change return value around points
        where both input values are the same. This allows implementations to
        limit the number of necessary point evaluations to a finite number.

        :param cmp: Comparator used to determine the set.
        :type cmp: Callable[[float, float], bool]
        :param other: Value or other signed measure to compare to.
        :type other: float | :class:`SignedMeasure`
        :return: Similarity class object representing the desired level set.
        :rtype: :class:`SimilarityClass`
        :raise NotImplementedError: `other` is a :class:`SignedMeasure` and
            the operation is not supported for that type of signed measure.

        .. note::

            As a general rule, this operation must be implemented for fixed
            levels, but need not be implemented for situations where `other`
            is a :class:`SignedMeasure`. As a fallback, the latter can be
            emulated by comparing the difference measure to the fixed level
            `0.0`.
        '''
        pass

    def __mul__(self, other) -> 'SignedMeasure':
        if isinstance(other, float):
            return self.scale(other)
        raise NotImplementedError()

    def __add__(self, other) -> 'SignedMeasure':
        if isinstance(other, SignedMeasure):
            return self.add(other)
        raise NotImplementedError()

    def __sub__(self, other) -> 'SignedMeasure':
        if isinstance(other, SignedMeasure):
            return self.sub(other)
        raise NotImplementedError()

    def __eq__(self, other) -> SimilarityClass:
        if isinstance(other, (float, SignedMeasure)):
            try:
                return self.level_set(operator.eq, other)
            except NotImplementedError:
                return (self - other).level_set(operator.eq, 0.0)
        raise NotImplementedError()

    def __ge__(self, other) -> SimilarityClass:
        if isinstance(other, (float, SignedMeasure)):
            try:
                return self.level_set(operator.ge, other)
            except NotImplementedError:
                return (self - other).level_set(operator.ge, 0.0)
        raise NotImplementedError()

    def __le__(self, other) -> SimilarityClass:
        if isinstance(other, (float, SignedMeasure)):
            try:
                return self.level_set(operator.le, other)
            except NotImplementedError:
                return (self - other).level_set(operator.le, 0.0)
        raise NotImplementedError()

    def __lt__(self, other) -> SimilarityClass:
        if isinstance(other, (float, SignedMeasure)):
            try:
                return self.level_set(operator.lt, other)
            except NotImplementedError:
                return (self - other).level_set(operator.lt, 0.0)
        raise NotImplementedError()

    def __gt__(self, other) -> SimilarityClass:
        if isinstance(other, (float, SignedMeasure)):
            try:
                return self.level_set(operator.gt, other)
            except NotImplementedError:
                return (self - other).level_set(operator.gt, 0.0)
        raise NotImplementedError()


@dataclass(frozen=True)
class SimilaritySpace:
    '''
    Simple description of a similarity space.
    '''
    measure: float
    empty: SimilarityClass
    universe: SimilarityClass
