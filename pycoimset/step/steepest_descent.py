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

from typing import Optional, cast

import numpy

from ..space import SignedMeasure, SimilarityClass
from .step_finder import StepFinder


class SteepestDescentStepFinder(StepFinder):
    '''
    The steepest descent step finder.

    This is the simplest and most straightforward step finder. By selecting
    the points with the lowest gradient density values, it achieves the
    highest descent per unit of step measure and is therefore named the
    "steepest descent" step finder.
    '''
    _step: Optional[list[SimilarityClass]]
    _error: Optional[float]

    def __init__(self):
        super().__init__()

        self._step = None
        self._error = None

    def find_step(self, radius: float, tol: float):
        # Check the two parameters.
        if radius <= 0.0:
            raise ValueError('radius')
        if tol <= 0.0:
            raise ValueError('tol')

        # Ensure that there are no `None` entries in the input.
        if any((grad is None for grad in self._grad)):
            raise ValueError('grad')
        grad_list = cast(list[SignedMeasure], self._grad)

        # Begin by obtaining the full step as an upper bound
        ub_set = [grad < 0.0 for grad in grad_list]
        ub_meas = sum((step.measure() for step in ub_set))

        # If the full step does not violate the radius, use it.
        if ub_meas <= radius:
            self._step = ub_set
            self._error = 0.0
            return

        # Find initial bounds.
        lb_lvl, ub_lvl = -tol, 0.0
        lb_set = [grad <= lb_lvl for grad in grad_list]
        while (lb_meas := sum((step.measure() for step in lb_set))) > radius:
            lb_lvl, ub_lvl, ub_set, ub_meas = 2 * lb_lvl, lb_lvl, lb_set, \
                lb_meas
            lb_set = [grad <= lb_lvl for grad in grad_list]

        # Main bisection loop.
        while ub_lvl - lb_lvl > tol / (2 * radius):
            mid_lvl = (ub_lvl + lb_lvl) / 2
            mid_set = [grad <= mid_lvl for grad in grad_list]

            if (meas := sum((step.measure() for step in mid_set))) <= radius:
                lb_lvl, lb_set, lb_meas = mid_lvl, mid_set, meas
            else:
                ub_lvl, ub_set, ub_meas = mid_lvl, mid_set, meas

        # Find the base step and calculate the size margins.
        step = lb_set
        meas = lb_meas

        min_size = radius - (tol - radius * (ub_lvl - lb_lvl) / abs(lb_lvl)) \
            - meas
        max_size = radius - meas

        # If the base step is too small, then we proceed to fill it out with
        # filler sets. We apportion the filler set to the components based on
        # the relative measures of the residual sets.
        if min_size > 0.0:
            res_set = [ub - lb for lb, ub in zip(lb_set, ub_set)]
            res_meas = [step.measure() for step in res_set]
            cum_meas = numpy.flip(numpy.cumsum(numpy.flip(res_meas)))

            fill_set = []
            fill_meas = 0.0
            for rem_tot, cur_meas, cur_step, cur_grad \
                    in zip(cum_meas, res_meas, res_set, grad_list):
                rem_ratio = cur_meas / rem_tot
                cur_fill = cur_step.choose_subset(
                    (min_size - fill_meas) * rem_ratio,
                    (max_size - fill_meas) * rem_ratio,
                    hint=cur_grad
                )
                fill_set.append(cur_fill)
                fill_meas += cur_fill.measure()

            step = [base | filler for base, filler in zip(step, fill_set)]
            meas += fill_meas

        # Save the step and calculate the error bound.
        self._step = step
        self._error = \
            (ub_lvl - lb_lvl) * radius \
            + abs(lb_lvl) * (radius - meas)

    @StepFinder.step.getter
    def step(self) -> list[SimilarityClass]:
        if self._step is None:
            raise ValueError('step')
        return self._step

    @StepFinder.error.getter
    def error(self) -> float:
        if self._error is None:
            raise ValueError('error')
        return self._error
