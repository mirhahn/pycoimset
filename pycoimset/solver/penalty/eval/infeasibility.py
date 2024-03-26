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
Infeasibility evaluator for the penalty method.
'''


from collections.abc import Callable
import math
from typing import TypeVar

import numpy
from numpy.typing import ArrayLike

from ....typing.space import SimilarityClass, SimilaritySpace
from ....util.controlled_eval import controlled_eval


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def eval_infeas(
    con_eval: list[
        Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]
    ],
    arg: SimilarityClass[_Tspc],
    rel_margin: float,
    tol: float,
    weights: ArrayLike | None = None,
    *,
    err_bnd: float = math.inf,
    err_decay: float = 0.5
) -> tuple[float, float]:
    '''
    Evaluate infeasibility.
    '''
    # Process weights
    if weights is None:
        weights = numpy.ones(len(con_eval), dtype=float)
    else:
        weights = numpy.broadcast_to(weights, len(con_eval))
    weights = weights.clip(min=0.0)

    # Catch edge case where all weights are zero
    if (weight_sum := weights.sum()) == 0.0:
        weights = numpy.ones(len(con_eval), dtype=float)
        weight_sum = weights.sum()

    weights = numpy.asarray(weights / weight_sum)

    # Bound oracle
    def bound_oracle(infeas: float):
        return rel_margin * max(tol, infeas - tol)

    # Inner evaluator
    def inner_eval(err_bnd: float):
        r = [func(arg, err_bnd * weight) for func, weight in zip(con_eval,
                                                                 weights)]
        return (sum((max(0, v) for v, _ in r)),
                sum((min(e, max(0, v + e)) for v, e in r)))

    # Perform controlled evaluation
    return controlled_eval(inner_eval, bound_oracle, err_bnd=err_bnd,
                           err_decay=err_decay)
