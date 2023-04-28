#
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
#
'''
Base classes for differentiable set functionals.
'''

from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional

from .space import SignedMeasure, SimilarityClass

__all__ = ['Functional']


class Functional(ABC):
    '''
    Abstract base class for a differentiable set functional.

    All set functionals in PyCoimset are differentiable. This class defines
    the minimal interface through which PyCoimset evaluates set functionals
    and retrieves their gradients.

    Set functionals can have multiple input variables. The precise number of
    inputs must be specified at the time of construction. All inputs will be
    specified prior to evaluation.

    :param num_inputs: Number of input variables.
    :type num_inputs: int
    :raise ValueError: :code:`num_inputs` is negative.
    '''
    _nin: int
    _tol: dict[Literal['val', 'grad.l1', 'grad.linf'], float]

    def __init__(self, num_inputs: int):
        super().__init__()

        if num_inputs < 0:
            raise ValueError('num_inputs')

        self._nin = num_inputs
        self._tol = {}

    @property
    def num_inputs(self) -> int:
        '''Number of input variables.'''
        return self._nin

    @property
    @abstractmethod
    def satisfies_linf_tolerance(self) -> bool:
        '''
        Indicates whether the current gradient satisfies the Linf error bound.

        Pointwise error bounds are much harder to satisfy and may not always
        be enforceable. Therefore, their satisfaction is not necessary for the
        optimization loop. However, the knowledge that an Linf error bound was
        met can be used to reduce re-evaluations of the gradient.

        After each gradient evaluation, this flag should be `True` if and only
        if an Linf error bound was set and was met.
        '''
        pass

    @property
    @abstractmethod
    def val(self) -> Optional[float]:
        '''
        Return functional value after evaluation.

        :retval None: The functional has not yet been evaluated.
        '''
        pass

    @property
    @abstractmethod
    def grad(self) -> Optional[list[SignedMeasure]]:
        '''
        Return gradient after evaluation.

        The result is returned as a list with one gradient measure per input.

        :retval None: The gradient has not yet been evaluated.
        '''
        pass

    @abstractmethod
    def set_input(self, val: SimilarityClass | Iterable[SimilarityClass],
                  idx: Optional[int | Iterable[int]] = None):
        '''
        Set input variables.

        This method can be used to set either a single input variable or
        multiple input variables at once.

        :param val: New input variable values. The functional may retain
            either references or shallow copies of these objects, so they
            should not be modified until they are replaced.
        :type val: :class:`SimilarityClass` |
            Iterable[:class:`SimilarityClass`]
        :param idx: Input variable indices to replace. This can only be a
            single value if :code:`val` is also a single object. If it is
            an iterable, it must be the same size as :code:`val`. If omitted,
            it defaults to :code:`range(num_inputs)`.
        :type idx: int | Iterable[int], optional
        :raise ValueError: The dimensions of :code:`val` and :code:`idx` do
            not match.
        :raise IndexError: One of the indices is out of range.
        :raise TypeError: One of the similarity classes is not of a type that
            the functional can process.
        '''
        pass

    def set_tolerances(self,
                       val_tol: float,
                       grad_l1_tol: float,
                       grad_linf_tol: Optional[float] = None):
        '''
        Reset evaluation error tolerances.

        Error control is essential to the correctness of the optimization
        loop. Therefore, any evaluation of the functional value must satisfy
        the absolute value tolerance. Any gradient evaluation must satisfy the
        L1 gradient tolerance at minimum and, if possible and given, also the
        Linf gradient tolerance. Whether the latter is satisfied should be
        indicated by the `satisfies_linf_tolerance` property after gradient
        evaluation.

        :param val_tol: Absolute value tolerance.
        :type val_tol: float
        :param grad_l1_tol: L1 gradient tolerance. This is an upper bound on
            the absolute error of the gradient measure for any input.
        :param grad_linf_tol: Linf gradient tolerance. This is an upper bound
            on the ratio between the absolute error of the gradient measure
            and the measure of its input set.
        :raise ValueError: One of the error bounds is not strictly positive.
        '''
        if val_tol <= 0.0:
            raise ValueError('val_tol')
        if grad_l1_tol <= 0.0:
            raise ValueError('grad_l1_tol')
        if grad_linf_tol is not None and grad_linf_tol <= 0.0:
            raise ValueError('grad_linf_tol')

        self._tol['val'] = val_tol
        self._tol['grad.l1'] = grad_l1_tol
        if grad_linf_tol is not None:
            self._tol['grad.linf'] = grad_linf_tol
        elif 'grad.linf' in self._tol:
            del self._tol['grad.linf']

    @abstractmethod
    def eval(self):
        '''
        Evaluate the function value.

        This performs the main function evaluation routine and generates the
        function value. In doing so, it observes the tolerances previously
        set by the user.

        The function value can subsequently be retrieved through the `val`
        property.
        '''
        pass

    @abstractmethod
    def eval_grad(self):
        '''
        Evaluate the gradient measure.

        This performs the gradient evaluation routine and any prerequisites
        (such as main function evaluation) and generates the gradient measure.
        In doing so, it observes the error tolerances last set by the user.

        The gradient measure can subsequently be retrieved through the `grad`
        property. The `satisfies_linf_tolerance` property indicates whether
        the gradient measure satisfies the specified Linf error bound.
        '''
        pass
