# PyCoimset - Python library for optimization with set-valued variables.
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
"""
Implementations of the basic unconstrained optimization loop.
"""

from dataclasses import dataclass
import math
from typing import Any, Optional

from ..functional import Functional
from ..problem import Problem
from ..space import SimilarityClass, SignedMeasure
from ..step import StepFinder, SteepestDescentStepFinder

__all__ = ['UnconstrainedSolver']


class UnconstrainedSolver:
    """
    Unconstrained optimization loop.

    This is a barebones implementation of the controlled descent framework
    presented in Section 3.1 of the thesis.
    """

    @dataclass
    class Parameters:
        abstol: float = 1e-3
        sigma_low: float = 0.1
        sigma_high: float = 0.5
        eta_1: float = 0.1
        eta_2: float = 0.1
        tr_radius: float = math.inf

    # Key-to-index map for variables.
    _keys: dict[Any, int]

    # Variables
    _var: list[SimilarityClass]

    # Step finder
    _step: StepFinder

    # Parameter set
    _param: Parameters

    # Objective functional
    _objfunc: Functional

    # Input map for the objective functional
    _objmap: list[int]

    # Maximal step size.
    _maxstep: float

    def __init__(self, problem: Problem):
        if problem.objective is None:
            raise ValueError('problem.objective')
        if len(problem.constraints) > 0:
            raise ValueError('problem.constraints')

        self._keys = {}
        self._var = []
        for key, val in problem.variables.items():
            self._keys[key] = len(self._var)
            self._var.append(val.copy(True))
        self._maxstep = sum((var.space.measure for var in self._var))

        self._step = SteepestDescentStepFinder()
        self._param = UnconstrainedSolver.Parameters()

        self._objfunc = problem.objective.functional
        self._objmap = [
            self._keys[key] for key in problem.objective.input_map
        ]

    @property
    def variables(self) -> list[SimilarityClass]:
        '''List of all variables.'''
        return list(self._var)

    @property
    def solution(self) -> dict[Any, SimilarityClass]:
        '''Dictionary of solution variables with associated keys.'''
        return {key: self._var[idx] for key, idx in self._keys.items()}

    @property
    def parameters(self) -> Parameters:
        '''Parameter object.'''
        return self._param
