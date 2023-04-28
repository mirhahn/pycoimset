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
from typing import Any, List, Mapping, Optional

from .functional import Functional
from .space import SimilarityClass


__all__ = [
    'Constraint',
    'MappedFunctional',
    'Problem',
]


@dataclass
class MappedFunctional:
    '''
    Functional with associated input mapping.
    '''
    functional: Functional
    input_map: list[Any]


@dataclass
class Constraint:
    '''
    Description of a differentiable scalar-valued constraint.
    '''
    val: MappedFunctional
    lb: Optional[float]
    ub: Optional[float]


class Problem:
    '''
    Description of an optimization problem.
    '''
    _var: dict[Any, SimilarityClass]
    _obj: Optional[MappedFunctional]
    _con: list[Constraint]

    def __init__(self):
        self._var = {}
        self._obj = None
        self._con = []

    def add_var(self, key: Any, initial_value: SimilarityClass):
        if key in self._var:
            raise KeyError(key)
        self._var[key] = initial_value

    def add_constr(self, constr: Constraint):
        if any(((missing := key) not in self._var
                for key in constr.val.input_map)):
            raise KeyError(str(missing))
        self._con.append(constr)

    @property
    def objective(self) -> Optional[MappedFunctional]:
        return self._obj

    @objective.setter
    def objective(self, value: Optional[MappedFunctional]):
        if value is not None and any(((missing := key) not in self._var
                                      for key in value.input_map)):
            raise KeyError(str(missing))
        self._obj = value

    @property
    def variables(self) -> Mapping[Any, SimilarityClass]:
        return self._var

    @property
    def constraints(self) -> List[Constraint]:
        return self._con
