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
Evaluator for the penalty functional value.
'''


from collections.abc import Callable
import math
from typing import TypeVar

import numpy
from numpy.typing import ArrayLike

from ....typing.space import SimilarityClass, SimilaritySpace
from ....util.controlled_eval import controlled_eval


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def eval_pen_summand(
    con_eval: Callable[[SimilarityClass[_Tspc], float], tuple[float, float]],
    arg: SimilarityClass[_Tspc],
    pen_param: float,
    pen_err_bnd: float,
    *,
    val_err_bnd: float = math.inf,
    val_err_decay: float = 0.5
) -> tuple[float, float]:
    '''
    Evaluate a single summand of the penalty function.

    Arguments
    ---------
    con_eval : (SimilarityClass[S], float) -> (float, float)
        Simplified evaluator for the constraint function. Receives an argument
        and an error bound and returns the evaluate and an error estimator.
        The error estimator must not exceed the error bound.

    arg : SimilarityClass[S]
        Argument at which to evaluate.

    pen_param : float
        Penalty parameter. Must be strictly postive.

    pen_err_bnd : float
        Output error bound. Must be strictly positive.

    val_err_bnd : float (optional, keyword-only)
        Initial error bound for the functional evaluation loop. Must be strictly
        positive. Can be infinite. Defaults to positive infinity.

    val_err_decay : float (optional, keyword-only)
        Error decay rate for the functional evaluation loop. Must be strictly
        between `0` and `1`. Defaults to `0.5`.

    Returns
    -------
    val : float
        Approximate value of the penalty summand.

    err : float
        Error estimate for the approximate penalty summand value.
    '''
    def inner_eval(err_bnd: float) -> tuple[float, float]:
        val, err = con_eval(arg, err_bnd)
        return max(0, val), min(err, max(0, val + err))

    def bound_func(val: float) -> float:
        return math.sqrt(val**2 + 2 * pen_err_bnd / pen_param)

    val, err = controlled_eval(inner_eval, bound_func, err_bnd=val_err_bnd,
                               err_decay=val_err_decay)

    return (pen_param / 2) * val**2, (pen_param / 2) * err * (2 * val + err)


def eval_pen_func(
    func_eval: list[Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]],
    weights: ArrayLike | None = None,
    *,
    err_bnd: float = math.inf,
    err_decay: float = 0.5
) -> Callable[[float], Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]]:
    '''
    Penalty functional evaluator.

    Arguments
    ---------
    func_eval : list of (SimilarityClass[S], float) -> (float, float)
        List of functional evaluators. Must not be empty. Last element is
        assumed to be the objective functional.

    weights : array-like of float (optional)
        Error apportionment weights. If not provided, all functionals have the
        same apportionment weight. If provided, must be broadcastable to the
        shape of `func_eval`. Negative weights are clamped to `0`. At least
        one weight must be strictly positive. If any entry is zero, then the
        evaluator must be able to function with an error bound of zero.

    err_bnd : float
        Initial error bound for evaluations. Is apportioned according to
        weights. Must be strictly positive. Can be infinite. Defaults to
        positive infinity.

    err_decay : float
        Error decay rate for evaluation loops. Must be strictly between `0`
        and `1`. Defaults to `0.5`.

    Returns
    -------
    (float) -> ((SimilarityClass[S], float) -> (float, float))
        Generator function that yields a simplified evaluator for the
        penalty functional at a fixed penalty parameter.
    '''
    # Bring weights into correct form.
    if weights is None:
        weights = numpy.ones(len(func_eval))
    else:
        weights = numpy.broadcast_to(numpy.asarray(weights), len(func_eval))
    weights = numpy.asarray(numpy.clip(weights, a_min=0.0, a_max=None, dtype=float))

    def fix_pen_param(
        pen_param: float
    ) -> Callable[
        [SimilarityClass[_Tspc], float],
        tuple[float, float]
    ]:
        # Write coefficients for error apportionment.
        bnd_coeff = numpy.copy(weights)
        bnd_coeff[:-1] *= pen_param
        bnd_coeff /= bnd_coeff.sum()

        def penalty_evaluator(arg: SimilarityClass[_Tspc], err_bnd: float) -> tuple[float, float]:
            if math.isinf(err_bnd):
                bnd = numpy.full_like(bnd_coeff, numpy.inf)
            else:
                bnd = bnd_coeff * err_bnd

            results = [
                *(
                    eval_pen_summand(
                        func, arg, pen_param, comp_bnd,
                        val_err_bnd=comp_bnd,
                        val_err_decay=err_decay
                    )
                    for func, comp_bnd in zip(func_eval[:-1], bnd[:-1])
                ),
                func_eval[-1](arg, bnd[-1])
            ]
            return sum((v for v, _ in results)), sum((e for _, e in results))
        return penalty_evaluator
    return fix_pen_param
