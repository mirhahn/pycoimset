# PyCoimset - Python library for optimization with set-valued variables.
# Copyright 2025 Mirko Hahn
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
"""
Generic types that can be used with arbitrary vector spaces.
"""

from collections.abc import Callable
from typing import Self, TypeVar
from ..typing.space import SignedMeasure, SimilarityClass, SimilaritySpace


__all__ = ['EmptyClass', 'UniversalClass']


Spc = TypeVar('Spc', bound=SimilaritySpace)
Cls = TypeVar('Cls', bound=SimilarityClass)


class EmptyClass(SimilarityClass[Spc]):
    """
    Generic representation of the empty class.
    """

    def __init__(self, space: Spc):
        super().__init__()
        self._space = space

    @property
    def space(self) -> Spc:
        return self._space

    @property
    def measure(self) -> float:
        return 0.0

    def subset(self, meas_low: float, meas_high: float,
               hint: SignedMeasure[Spc] | None = None) -> Self:
        if meas_low > 0.0 or meas_high < 0.0:
            raise ValueError('cannot satisfy subset request')
        return self

    def __invert__(self) -> SimilarityClass[Spc]:
        return self._space.universal_class

    def __or__(self, b: Cls) -> Cls:
        return b

    def __ror__(self, a: Cls) -> Cls:
        return a

    def __and__(self, b: SimilarityClass[Spc]) -> Self:
        return self

    def __rand__(self, a: SimilarityClass[Spc]) -> Self:
        return self

    def __sub__(self, b: SimilarityClass[Spc]) -> Self:
        return self

    def __rsub__(self, a: Cls) -> Cls:
        return a

    def __xor__(self, b: Cls) -> Cls:
        return b

    def __rxor__(self, a: Cls) -> Cls:
        return a


class UniversalClass(SimilarityClass[Spc]):
    """
    Generic representation of the universal class.
    """

    def __init__(self, space: Spc, subset_func: Callable[
                    [float, float, SignedMeasure[Spc] | None],
                    SimilarityClass[Spc]
                 ]):
        super().__init__()
        self._space = space
        self._subset = subset_func

    @property
    def space(self) -> Spc:
        return self._space

    @property
    def measure(self) -> float:
        return self._space.measure

    def subset(self, meas_low: float, meas_high: float,
               hint: SignedMeasure[Spc] | None = None
               ) -> SimilarityClass[Spc]:
        return self._subset(meas_low, meas_high, hint)

    def __invert__(self) -> SimilarityClass[Spc]:
        return self._space.empty_class

    def __or__(self, b: SimilarityClass[Spc]) -> Self:
        return self

    def __ror__(self, a: SimilarityClass[Spc]) -> Self:
        return self

    def __and__(self, b: Cls) -> Cls:
        return b

    def __rand__(self, a: Cls) -> Cls:
        return a

    def __sub__(self, b: SimilarityClass[Spc]) -> SimilarityClass[Spc]:
        return ~b

    def __rsub__(self, a: SimilarityClass[Spc]) -> SimilarityClass[Spc]:
        return self._space.empty_class

    def __xor__(self, b: SimilarityClass[Spc]) -> SimilarityClass[Spc]:
        return ~b

    def __rxor__(self, a: SimilarityClass[Spc]) -> SimilarityClass[Spc]:
        return ~a
