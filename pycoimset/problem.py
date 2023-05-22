#
# PyCoimset library for optimization with set-valued variables.
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
Problem specification classes.
'''

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, TypeVar

from .typing import Functional, SimilarityClass, SimilaritySpace


__all__ = [
    'Constraint',
    'Problem',
]


Spc = TypeVar('Spc', bound=SimilaritySpace)


@dataclass
class Constraint(Generic[Spc]):
    '''
    Description of a differentiable scalar-valued constraint.
    '''
    val: Functional[Spc]
    lb: Optional[float] = None
    ub: Optional[float] = None


class Problem(Generic[Spc]):
    '''
    Description of an optimization problem.
    '''
    _spc: Spc
    _obj: Functional[Spc]
    _con: tuple[Constraint[Spc], ...]
    _x0: Optional[SimilarityClass[Spc]]

    def __init__(self, space: Spc, objective: Functional[Spc],
                 *constraints: Constraint[Spc],
                 initial_value: Optional[SimilarityClass[Spc]] = None):
        self._spc = space
        self._obj = objective
        self._con = constraints
        self._x0 = initial_value

    @property
    def space(self) -> Spc:
        '''Variable space.'''
        return self._spc

    @property
    def initial_value(self) -> SimilarityClass:
        '''Initial value.'''
        if self._x0 is None:
            return self._spc.empty_class
        else:
            return self._x0

    @property
    def objective(self) -> Functional[Spc]:
        return self._obj

    @property
    def constraints(self) -> Sequence[Constraint[Spc]]:
        return self._con
