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
'''
Steepest descent step finding.
'''

import math
from typing import Optional, TypeVar

from ..typing import (
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)


Spc = TypeVar('Spc', bound=SimilaritySpace)


class SteepestDescentStepFinder(UnconstrainedStepFinder[Spc]):
    '''
    The steepest descent step finder.

    This is the simplest and most straightforward step finder. By selecting
    the points with the lowest gradient density values, it achieves the
    highest descent per unit of step measure and is therefore named the
    "steepest descent" step finder.
    '''
    _grad: Optional[SignedMeasure[Spc]]
    _radius: float
    _tol: float
    _result: Optional[tuple[SimilarityClass[Spc], float]]

    def __init__(self):
        self._grad = None
        self._radius = math.inf
        self._tol = math.inf
        self._result = None

    @property
    def quality(self) -> float:
        return 1.0

    @property
    def gradient(self) -> Optional[SignedMeasure[Spc]]:
        return self._grad

    @gradient.setter
    def gradient(self, grad: Optional[SignedMeasure[Spc]]) -> None:
        self._grad = grad
        self._result = None

    @property
    def tolerance(self) -> float:
        return self._tol

    @tolerance.setter
    def tolerance(self, tol: float) -> None:
        if tol <= 0.0:
            raise ValueError('tolerance must be strictly positive')
        self._tol = tol
        if self._result is not None and self._result[1] > self._tol:
            self._result = None

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, r: float) -> None:
        if r <= 0.0:
            raise ValueError('trust-region radius must be strictly positive')
        self._radius = r
        if self._result is not None and (self._result[0].measure
                                         > self._radius):
            self._result = None

    def get_step(self) -> tuple[SimilarityClass[Spc], float]:
        # Return cached result if available.
        if self._result is not None:
            return self._result

        # Ensure that there are no `None` entries in the input.
        if (grad := self._grad) is None:
            raise ValueError('`gradient` is not set')

        # Ensure that tolerance is finite.
        if math.isinf(tol := self._tol):
            raise ValueError('`tolerance` is not set')

        # Copy trust-region radius into local namespace.
        radius = self._radius

        # Begin by obtaining the full step as an upper bound
        ub = grad < 0.0

        # If the full step does not violate the radius, use it.
        if ub.measure <= radius:
            self._result = (ub, 0.0)
            return self._result

        # Find initial bounds.
        lb_lvl, ub_lvl = -self._tol, 0.0
        lb = grad <= lb_lvl
        while lb.measure > radius:
            lb_lvl, ub_lvl, ub = 2 * lb_lvl, lb_lvl, lb
            lb = grad <= lb_lvl

        # Main bisection loop.
        while ub_lvl - lb_lvl > tol / (2 * radius):
            mid_lvl = (ub_lvl + lb_lvl) / 2
            mid = grad <= mid_lvl

            if mid.measure <= radius:
                lb_lvl, lb = mid_lvl, mid
            else:
                ub_lvl, ub = mid_lvl, mid

        # Find the base step and calculate the size margins.
        step = lb
        step_measure = step.measure

        max_size = radius - step_measure
        min_size = max_size - (tol - radius * (ub_lvl - lb_lvl)) / abs(lb_lvl)

        # If the base step is too small, then we proceed to fill it out with
        # filler sets. We apportion the filler set to the components based on
        # the relative measures of the residual sets.
        if min_size > 0.0:
            step = step | (ub - lb).subset(min_size, max_size, hint=grad)

        # Save the step and calculate the error bound.
        self._result = (step, (ub_lvl - lb_lvl) * radius + abs(lb_lvl)
                        * (radius - step.measure))
        return self._result
