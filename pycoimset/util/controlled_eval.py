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
Generic controlled evaluation loop.
'''


from collections.abc import Callable
import math
from typing import TypeVar


_T = TypeVar('_T')


def controlled_eval(
    eval_func: Callable[[float], tuple[_T, float]],
    param_func: Callable[[_T], float],
    *,
    err_bnd: float = math.inf,
    err_decay: float = 0.5
) -> tuple[_T, float]:
    '''
    Controlled evaluation loop.

    Arguments
    ---------
    eval_func : (float) -> (T, float)
        Evaluation function. Accepts error bound and parameters and returns
        evaluate and error estimator. Error bound may be infinite. Error
        estimator must be finite and no larger than error bound.

    param_func : (T) -> (float)
        Parameter update function. Accepts evaluate and parameters and returns
        new error bound and new parameters. There must be a guarantee that
        for evaluates with an error estimate below a fixed, strictly positive
        threshold, then the new error bound remains above the error estimate
        and the parameters remain unchanged.

    err_bnd : float (optional, keyword-only)
        Initial error bound. Must be strictly positive. Can be infinite to
        indicate a desire for evaluation without specific error control.
        Defaults to positive infinity.

    err_decay : float (optional, keyword-only)
        Error decay rate. Must be strictly between `0` and `1`. Defaults to
        `0.5`.

    Returns
    -------
    x : T
        Output value.

    e : float
        Error estimate. Must be non-negative and finite.

    Raises
    ------
    ValueError
        One of the argument constraints has been violated.
    '''
    # Check inputs
    if err_bnd <= 0.0:
        raise ValueError('Initial error bound must be strictly positive.')
    if err_decay <= 0.0 or err_decay >= 1.0:
        raise ValueError(
            'Error decay rate must be strictly between 0.0 and 1.0.'
        )

    # Initial evaluation
    val, err = eval_func(err_bnd)
    next_bnd = param_func(val)

    # Improvement loop
    while err > next_bnd:
        # FIXME: This cast should not be necessary. Type checker does weird
        # stuff here.
        err_bnd = min(err_bnd, err_decay * next_bnd)
        val, err = eval_func(err_bnd)
        next_bnd = param_func(val)

    # Return results
    return val, err
