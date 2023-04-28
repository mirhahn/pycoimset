# PyCoimset - Python library for optimization with set-valued constraints.
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
Generic base classes.
'''


from abc import ABC, abstractmethod
from typing import Collection, Optional
import typing

from ..space import SignedMeasure, SimilarityClass

__all__ = ['StepFinder']


class StepFinder(ABC):
    _grad: list[Optional[SignedMeasure]]

    def __init__(self):
        self._grad = []

    def set_grad(self, grad: SignedMeasure | Collection[SignedMeasure],
                 idx: Optional[int | Collection[int]] = None):
        '''
        Set input gradients.
        '''
        # Convert everything into sized iterables.
        if isinstance(grad, SignedMeasure):
            grad = [grad]
        if idx is None:
            idx = range(len(grad))
        elif isinstance(idx, int):
            idx = [idx]

        # Extend gradient list if necessary.
        max_idx = max(idx)
        if max_idx >= len(self._grad):
            self._grad.extend([None] * (max_idx + 1 - len(self._grad)))

        # Enter new gradients into list.
        for i, g in zip(idx, grad):
            self._grad[i] = g

    @abstractmethod
    def find_step(self, radius: float, tol: float):
        '''
        Find a descent step.

        This performs the main calculation of the step. After this method
        returns, the property `step` should return a list with steps for each
        variable whose gradient was given.

        :param radius: Trust region radius. The cumulative measure of all
            components of the step must not be greater than this number.
        :type radius: float
        :param tol: Error tolerance. This is the maximal margin by which the
            cumulative gradient measure of the step components may exceed the
            product between the trust region radius and the mean of the
            gradient density over the support of its negative part.
        :type tol: float

        :raise ValueError: Either `radius` or `tol` are not strictly positive.
        '''
        pass

    @property
    @abstractmethod
    def step(self) -> list[SimilarityClass]:
        '''
        Steps determined by the last call to `find_step`.

        The individual step components are presented as a list in the same
        order as their corresponding gradient measures in the input.

        :raise ValueError: There was no prior call to `find_step`.
        '''
        pass

    @property
    @abstractmethod
    def error(self) -> float:
        '''
        Upper bound on the error made by the last call to `find_step`.

        :raise ValueError: There was no prior call to `find_step`.
        '''
        pass
