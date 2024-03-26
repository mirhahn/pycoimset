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
Penalty gradient evaluators.
'''


from collections.abc import Callable
import math
from typing import Final, Literal, TypeVar

import numpy
from numpy.typing import ArrayLike

from ....typing.functional import ErrorNorm
from ....typing.space import SignedMeasure, SimilarityClass, SimilaritySpace
from ....util.cache import error_control_cache
from ....util.controlled_eval import controlled_eval


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def eval_pen_grad_summand(
    con_eval: Callable[[SimilarityClass[_Tspc], float], tuple[float, float]],
    con_grad_eval: Callable[[SimilarityClass[_Tspc], float], tuple[SignedMeasure[_Tspc], float]],
    arg: SimilarityClass[_Tspc],
    pen_param: float,
    pen_grad_err_bnd: float,
    *,
    err_norm: ErrorNorm,
    grad_err_bnd: float = math.inf,
    grad_err_decay: float = 0.5,
    val_err_decay: float = 0.1,
    val_denom_eps: float = 0.1
) -> tuple[SignedMeasure[_Tspc], float]:
    '''
    Evaluate gradient of single penalty function summand.

    Arguments
    ---------
    con_eval : (SimilarityClass[S], float) -> (float, float)
        Simplified evaluator for the constraint functional value.

    con_grad_eval : (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        Simplified evaluator for the constraint functional gradient.

    arg : SimilarityClass[S]
        Point at which to evaluate the gradient.

    pen_param : float
        Penalty parameter. Must be non-negative.

    pen_grad_err_bnd : float
        Gradient error bound. Must be strictly positive or non-negative if
        all subordinate evaluators accept zero as an error bound.

    err_norm : ErrorNorm (keyword-only)
        Error control norm for the constraint functional. Must be congruent
        with the gradient control error norm used by `con_grad_eval`.

    grad_err_bnd : float (optional, keyword-only)
        Initial error bound for the constraint functional gradient. Passed on
        to the evaluation loop as an initial bound. Must be strictly positive
        unless all evaluators return an error estimate of zero. Can be infinite.
        Defaults to positive infinity.

    grad_err_decay: float (optional, keyword-only)
        Error decay rate for the controlled evaluation loop. Must be strictly
        between `0` and `1`. Defaults to `0.5`.

    val_err_decay : float (optional, keyword-only)
        Tuning parameter. Must be strictly between `0` and `1`. Controls the
        fraction of the actual upper error bound on the functional value that
        is passed as an effective error bound to the evaluator. Defaults to
        `0.1`.

    val_denom_eps : float (optional, keyword-only)
        Tuning parameter. Must be strictly positive. Used to clamp the
        denominator of the value error bound to a strictly positive value.
        Defaults to `0.1`.

    Returns
    -------
    grad : SignedMeasure[S]
        Approximate gradient measure.

    err : float
        Error estimator for the approximate gradient measure. Uses `err_norm`.
    '''
    norm_type: Final[Literal['L1', 'Linfty']] = \
        'L1' if err_norm is ErrorNorm.L1 \
        else 'Linfty'

    def inner_eval(err_bnd: float) -> tuple[tuple[SignedMeasure[_Tspc], float, float], float]:
        grad, err_grad = con_grad_eval(arg, err_bnd)
        val, err_val = con_eval(arg, val_err_decay * pen_grad_err_bnd / max(val_denom_eps, pen_param * grad.norm(norm_type)))
        return (grad, val, err_val), err_grad

    def bound_func(val: tuple[SignedMeasure[_Tspc], float, float]) -> float:
        grad, func, err_func = val
        denom = pen_param * (max(0, func) + err_func)
        if denom > 0:
            return (grad_err_bnd - pen_param * err_func * grad.norm(norm_type)) / denom
        else:
            return math.inf

    val, err_grad = controlled_eval(inner_eval, bound_func,
                                    err_bnd=grad_err_bnd,
                                    err_decay=grad_err_decay)
    grad, func, err_func = val
    return pen_param * max(0, func) * grad, pen_param * (
        (max(0, func) + err_func) * err_grad + err_func * grad.norm(norm_type)
    )


def eval_pen_grad(
    func_eval: list[Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]],
    grad_eval: list[Callable[[SimilarityClass[_Tspc], float], tuple[SignedMeasure[_Tspc], float]]],
    err_norm: ErrorNorm,
    weights: ArrayLike | None = None,
    *,
    grad_err_bnd: float = math.inf,
    grad_err_decay: float = 0.5,
    val_err_decay: float = 0.1,
    val_denom_eps: float = 0.1,
    cache_size: int = 2
) -> Callable[[float], Callable[[SimilarityClass[_Tspc], float], tuple[SignedMeasure[_Tspc], float]]]:
    '''
    Create a penalty gradient evaluator.

    Arguments
    ---------
    func_eval : list of (SimilarityClass[S], float) -> (float, float)
        List of functional evaluators. Must not be empty. Last element is
        assumed to be the objective functional.

    grad_eval : list of (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        List of gradient evaluators. Must correspond element-wise with
        `func_eval`.

    err_norm : ErrorNorm
        Joint error control norm for the gradient evaluators.

    weights : array-like of float (optional)
        Weight values controlling error bound apportionment. If not provided,
        all summands are apportioned the same error budget. If provided, must
        be broadcastable to the length of `func_eval`.

    grad_err_bnd : float (optional, keyword-only)
        Initial error bound for the constraint functional gradient. Passed on
        to the evaluation loop as an initial bound. Must be strictly positive
        unless all evaluators return an error estimate of zero. Can be infinite.
        Defaults to positive infinity.

    grad_err_decay: float (optional, keyword-only)
        Error decay rate for the controlled evaluation loop. Must be strictly
        between `0` and `1`. Defaults to `0.5`.

    val_err_decay : float (optional, keyword-only)
        Tuning parameter. Must be strictly between `0` and `1`. Controls the
        fraction of the actual upper error bound on the functional value that
        is passed as an effective error bound to the evaluator. Defaults to
        `0.1`.

    val_denom_eps : float (optional, keyword-only)
        Tuning parameter. Must be strictly positive. Used to clamp the
        denominator of the value error bound to a strictly positive value.
        Defaults to `0.1`.

    cache_size: int (optional, keyword-only)
        Indicates that the evaluator produced for a fixed penalty parameter
        should have its own cache to mitigate the effort of adding signed
        measures multiple times. Defaults to `2` to reflect the need to undo
        a single step.

    Returns
    -------
    (float) -> ((SimilarityClass[S], float) -> (SignedMeasure, float))
        Function that returns a gradient evaluator for a fixed penalty
        parameter.
    '''
    # Bring weights into correct form.
    if weights is None:
        weights = numpy.ones(len(func_eval))
    else:
        weights = numpy.broadcast_to(numpy.asarray(weights), len(func_eval))
    weights = numpy.asarray(numpy.clip(weights, a_min=0.0, a_max=None, dtype=float))

    def fix_pen_param(pen_param: float) -> Callable[[SimilarityClass[_Tspc], float], tuple[SignedMeasure[_Tspc], float]]:
        # Write coefficients for error apportionment.
        bnd_coeff = numpy.copy(weights)
        bnd_coeff[:-1] *= pen_param
        bnd_coeff /= bnd_coeff.sum()

        def penalty_evaluator(arg: SimilarityClass[_Tspc], err_bnd: float) -> tuple[SignedMeasure[_Tspc], float]:
            if math.isinf(err_bnd):
                bnd = numpy.full_like(bnd_coeff, numpy.inf)
            else:
                bnd = bnd_coeff * err_bnd
            if math.isinf(err_bnd) or math.isinf(grad_err_bnd):
                grad_bnd = numpy.full_like(bnd_coeff, numpy.inf)
            else:
                grad_bnd = bnd_coeff * grad_err_bnd

            results = [
                *(
                    eval_pen_grad_summand(
                        func, grad, arg, pen_param, comp_bnd,
                        err_norm=err_norm,
                        grad_err_bnd=comp_grad_bnd,
                        grad_err_decay=grad_err_decay
                    )
                    for func, grad, comp_bnd, comp_grad_bnd in zip(
                        func_eval[:-1], grad_eval[:-1], bnd[:-1], grad_bnd[:-1]
                    )
                ),
                grad_eval[-1](arg, bnd[-1])
            ]
            return sum((v for v, _ in results[1:]), start=results[0][0]), sum((e for _, e in results))
        return error_control_cache(penalty_evaluator, cache_size=2)
    return fix_pen_param
