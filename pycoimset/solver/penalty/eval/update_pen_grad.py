'''
Entangled penalty and gradient update loop.
'''

from collections.abc import Callable
import math
from typing import TypeVar

from ...unconstrained.eval import eval_grad, eval_instat
from ....typing.functional import ErrorNorm
from ....typing.space import SignedMeasure, SimilarityClass, SimilaritySpace


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def update_pen_grad(
    fix_pen_param: Callable[
        [float],
        Callable[
            [SimilarityClass[_Tspc], float],
            tuple[SignedMeasure[_Tspc], float]
        ]
    ],
    err_norm: ErrorNorm,
    arg: SimilarityClass[_Tspc],
    tol_infeas: float,
    tol_instat: float,
    tol_instat_relaxed: float,
    margin_instat: float,
    margin_step: float,
    margin_proj_desc: float,
    step_qual: float,
    infeas: float,
    pen_param: float,
    pen_param_limit: float,
    tr_rad: float,
    pen_grad: Callable[
        [SimilarityClass[_Tspc], float],
        tuple[SignedMeasure[_Tspc], float]
    ] | None = None,
    *,
    instat_evaluator: Callable[
        [SignedMeasure[_Tspc]],
        tuple[SimilarityClass[_Tspc], float]
    ] | None = None,
    err_bnd: float = math.inf,
    err_decay: float = 0.5
) -> tuple[
    SignedMeasure[_Tspc], float,
    float, Callable[
        [SimilarityClass[_Tspc], float],
        tuple[SignedMeasure[_Tspc], float]
    ]
]:
    '''
    Update penalty parameter and evaluate gradient.
    '''
    # Generate penalty gradient evaluator if none is provided.
    if pen_grad is None:
        pen_grad = fix_pen_param(pen_param)

    # Make instationarity evaluator if necessary
    if instat_evaluator is None:
        instat_evaluator = eval_instat(arg.space)

    # Initial evaluation, short-circuit if feasible
    grad, _, tau, err = eval_grad(pen_grad, arg, tol_instat, tr_rad, margin_instat,
                                  margin_step, margin_proj_desc, step_qual, err_norm,
                                  instat_eval=instat_evaluator, err_bnd=err_bnd,
                                  err_decay=err_decay)
    if infeas <= tol_infeas:
        return grad, err, pen_param, pen_grad

    # Double at relaxed threshold
    if 2 * pen_param <= pen_param_limit and tau <= tol_instat_relaxed:
        pen_param = 2 * pen_param
        pen_grad = fix_pen_param(pen_param)
        grad, _, tau, err = eval_grad(grad=pen_grad,
                                      set_cur=arg,
                                      instat_tol=tol_instat,
                                      tr_rad=tr_rad,
                                      xi_tau=margin_instat,
                                      xi_delta=margin_step,
                                      xi_g=margin_proj_desc,
                                      step_qual=step_qual,
                                      err_norm=err_norm,
                                      err_bnd=err_bnd,
                                      err_decay=err_decay,
                                      instat_eval=instat_evaluator)

    # Double repeatedly at hard threshold
    while pen_param <= pen_param_limit and tau <= tol_instat:
        pen_param = 2 * pen_param
        pen_grad = fix_pen_param(pen_param)
        grad, _, tau, err = eval_grad(grad=pen_grad,
                                      set_cur=arg,
                                      instat_tol=tol_instat,
                                      tr_rad=tr_rad,
                                      xi_tau=margin_instat,
                                      xi_delta=margin_step,
                                      xi_g=margin_proj_desc,
                                      step_qual=step_qual,
                                      err_norm=err_norm,
                                      err_bnd=err_bnd,
                                      err_decay=err_decay,
                                      instat_eval=instat_evaluator)
    
    return grad, err, pen_param, pen_grad
