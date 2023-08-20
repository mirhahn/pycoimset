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
Static typing protocols for solver components.
'''

from typing import Optional, Protocol, TypeVar

from .space import SignedMeasure, SimilarityClass, SimilaritySpace


__all__ = [
    'UnconstrainedStepFinder',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


class UnconstrainedStepFinder(Protocol[Spc]):
    '''Protocol for unconstrained step finders.'''

    @property
    def quality(self) -> float:
        '''Quality constant of this step finder.'''
        ...

    @property
    def gradient(self) -> Optional[SignedMeasure[Spc]]:
        '''Gradient measure.'''
        ...

    @gradient.setter
    def gradient(self, grad: Optional[SignedMeasure[Spc]]) -> None:
        ...

    @property
    def radius(self) -> float:
        '''Trust-region radius.'''
        ...

    @radius.setter
    def radius(self, r: float) -> None:
        ...

    @property
    def tolerance(self) -> float:
        '''Step finding tolerance.'''
        ...

    @tolerance.setter
    def tolerance(self, tol: float) -> None:
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
