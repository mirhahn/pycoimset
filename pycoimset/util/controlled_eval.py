'''
Generic controlled evaluation loop.
'''


from collections.abc import Callable
import math
from typing import TypeVar, TypeVarTuple, cast


_T = TypeVar('_T')
_P = TypeVarTuple('_P')


def controlled_eval(
    eval_func: Callable[[float, *_P], tuple[_T, float]],
    param_func: Callable[[_T, *_P], tuple[float, tuple[*_P]]],
    *param_init: *_P,
    err_bnd: float = math.inf,
    err_decay: float = 0.5
) -> tuple[_T, float, *_P]:
    '''
    Controlled evaluation loop.

    Arguments
    ---------
    eval_func : (float, *P) -> (T, float)
        Evaluation function. Accepts error bound and parameters and returns
        evaluate and error estimator. Error bound may be infinite. Error
        estimator must be finite and no larger than error bound.

    param_func : (T, *P) -> (float, *P)
        Parameter update function. Accepts evaluate and parameters and returns
        new error bound and new parameters. There must be a guarantee that
        for evaluates with an error estimate below a fixed, strictly positive
        threshold, then the new error bound remains above the error estimate
        and the parameters remain unchanged.

    param_init : *P
        Additional variadic arguments to be used as initial parameters for the
        evaluator and update function.

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

    p : *P
        Additional parameters.

    Raises
    ------
    ValueError
        One of the argument constraints has been violated.
    '''
    # Check inputs
    if err_bnd <= 0.0:
        raise ValueError('Initial error bound must be strictly positive.')
    if err_decay <= 0.0 or err_decay >= 1.0:
        raise ValueError('Error decay rate must be strictly between 0.0 and 1.0.')

    # Initial evaluation
    param = param_init
    val, err = eval_func(err_bnd, *param)
    next_bnd, next_param = param_func(val, *param)

    # Improvement loop
    while err > next_bnd or param != next_param:
        # FIXME: This cast should not be necessary. Type checker does weird
        # stuff here.
        param = cast(tuple[*_P], next_param)
        err_bnd = min(err_bnd, err_decay * next_bnd)
        val, err = eval_func(err_bnd, *param)
        next_bnd, next_param = param_func(val, *param)

    # Return results
    return val, err, *param
