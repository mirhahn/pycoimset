# PyCoimset: Python library for COntinuous IMprovement of SETs
#
# Copyright 2024 Mirko Hahn
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
Component evaluators for the penalty method.
'''


from collections.abc import Callable
import math
from typing import TypeVar

from ....typing.functional import ErrorNorm, Functional
from ....typing.space import SignedMeasure, SimilarityClass, SimilaritySpace
from ...unconstrained.eval import make_func_eval, make_grad_eval


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def adapt_grad_eval(
    grad_eval: Callable[
        [SimilarityClass[_Tspc], float],
        tuple[SignedMeasure[_Tspc], float]
    ],
    from_norm: ErrorNorm,
    to_norm: ErrorNorm
) -> Callable[
    [SimilarityClass[_Tspc], float],
    tuple[SignedMeasure[_Tspc], float]
]:
    '''
    Adapts the control norm of a gradient evaluator.

    Arguments
    ---------
    grad_eval : (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        Original gradient evaluator.

    from_norm : ErrorNorm
        Original control norm.

    to_norm : ErrorNorm
        Desired control norm.

    Returns
    -------
    (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        Modified gradient evaluator.
    '''
    if to_norm is from_norm:
        return grad_eval
    elif from_norm is ErrorNorm.Linfty and to_norm is ErrorNorm.L1:
        def wrapped_evaluator(arg: SimilarityClass[_Tspc], err_bnd: float
                              ) -> tuple[SignedMeasure[_Tspc], float]:
            spc_meas = arg.space.measure
            if spc_meas == 0.0:
                err_bnd = math.inf
            else:
                err_bnd /= spc_meas
            grad, err = grad_eval(arg, err_bnd)
            return grad, err * spc_meas
        return wrapped_evaluator
    else:
        raise ValueError(f'Cannot convert control norm from {from_norm} to {to_norm}')


def make_component_eval(
    obj: Functional[_Tspc],
    *cons: Functional[_Tspc],
    cache_size: int
) -> tuple[
    list[Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]],
    list[Callable[[SimilarityClass[_Tspc], float], tuple[SignedMeasure[_Tspc], float]]],
    ErrorNorm
]:
    '''
    Creates cached functional and gradient evaluators for all functionals.

    This is a helper that is used to generate all evaluators simultaneously.
    By generating all evaluators once and reusing them in multiple higher-order
    evaluators, cache can be shared between all evaluators.

    By convention, the objective is assigned the last place in the output lists.

    Arguments
    ---------
    obj : Functional[S]
        Objective functional.

    *cons : Functional[S]
        Constraint functionals.

    cache_size : int (keyword-only)
        Size of the LRU caches for each evaluator.

    Returns
    -------
    func_eval : list of (SimilarityClass[S], float) -> (float, float)
        List of simplified functional evaluators.

    grad_eval : list of (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        List of simplified gradient evaluators.

    err_norm : ErrorNorm
        Error control norm for all gradient evaluators.

    Notes
    -----
        If functionals occur multiple times, then the same cache will be used
        for each occurrence.
    '''
    funcs = [*cons, obj]
    func_eval = {}
    grad_eval = {}

    # Determine gradient control norm.
    err_norm = ErrorNorm.L1 if any((func.grad_tol_type is ErrorNorm.L1 for func in funcs)) \
        else ErrorNorm.Linfty

    for func in funcs:
        if id(func) not in func_eval:
            func_eval[id(func)] = make_func_eval(func, cache_size=cache_size)
            grad_eval[id(func)] = adapt_grad_eval(
                make_grad_eval(func, cache_size=cache_size),
                from_norm=func.grad_tol_type,
                to_norm=err_norm
            )
    return ([func_eval[id(func)] for func in funcs],
            [grad_eval[id(func)] for func in funcs],
            err_norm)
