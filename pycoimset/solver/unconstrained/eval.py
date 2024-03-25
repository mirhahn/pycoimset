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
Custom evaluators for the unconstrained optimization solver.
'''


from collections.abc import Callable
import math
from types import NotImplementedType
from typing import TypeVar, assert_never

from ...typing.functional import ErrorNorm, Functional
from ...typing.space import SignedMeasure, SimilarityClass, SimilaritySpace
from ...util.cache import cached_external_property, error_control_cache
from ...util.controlled_eval import controlled_eval


_Tspc = TypeVar('_Tspc', bound=SimilaritySpace)


def eval_instat(_: _Tspc, cache_size: int | None = 1
                ) -> Callable[[SignedMeasure[_Tspc]], tuple[SimilarityClass[_Tspc], float]]:
    '''
    Cached instationarity evaluator.

    Arguments
    ---------
    cache_size : int
        Maximal cache size. `None` indicates indefinite cache. Defaults to
        `1`.

    Returns
    -------
    SignedMeasure[S] -> (SimilarityClas[S], float)
        Callable that evaluates the strict sublevel set for level `0` and
        instationarity according to a given gradient measure.
    '''
    def inner_evaluator(grad: SignedMeasure[_Tspc]
                        ) -> tuple[SimilarityClass[_Tspc], float]:
        return (set_neg := grad < 0), abs(grad(set_neg))
    return cached_external_property(cache_size)(inner_evaluator)


def make_func_eval(func: Functional[_Tspc], *, cache_size: int = 2
                   ) -> Callable[[SimilarityClass[_Tspc], float], tuple[float, float]]:
    '''
    Create a simplified, possibly cached evaluator for a functional.

    Arguments
    ---------
    func : Functional[S]
        Set functional to be evaluated.

    cache_size : int (optional, keyword-only)
        Maximum size of the LRU cache. Defaults to `2`.

    Returns
    -------
    eval_func : (SimilarityClass[S], float) -> (float, float)
        Evaluator function that takes a functional argument and an error bound
        and returns an approximate evaluate and an error estimate.
    '''
    # Simple evaluator without cache
    def simple_eval(arg: SimilarityClass[_Tspc], err_bnd: float
                    ) -> tuple[float, float]:
        if arg is func.arg:
            func.val_tol = min(func.val_tol, err_bnd)
        else:
            func.arg = arg
            func.val_tol = err_bnd
        return func.get_value()
    return error_control_cache(simple_eval, cache_size=cache_size)


def make_grad_eval(
    func: Functional[_Tspc], *, cache_size: int = 2
) -> Callable[
    [SimilarityClass[_Tspc], float],
    tuple[SignedMeasure[_Tspc], float]
]:
    '''
    Create a simplified, possibly cached evaluator for a functional gradient.

    Arguments
    ---------
    func : Functional[S]
        Set functional to be evaluated.

    cache_size : int (optional, keyword-only)
        Maximum size of the LRU cache. Defaults to `2`.

    Returns
    -------
    eval_func : (SimilarityClass[S], float) -> (SignedMeasure[S], float)
        Evaluator function that takes a functional argument and an error bound
        and returns an approximate gradient and an error estimate.
    '''
    # Simple evaluator without cache
    def simple_eval(arg: SimilarityClass[_Tspc], err_bnd: float
                    ) -> tuple[SignedMeasure[_Tspc], float]:
        if arg is func.arg:
            func.grad_tol = min(func.grad_tol, err_bnd)
        else:
            func.arg = arg
            func.grad_tol = err_bnd
        return func.get_gradient()
    return error_control_cache(simple_eval, cache_size=cache_size)


def eval_rho(eval_func: Callable[
                [SimilarityClass[_Tspc], float],
                tuple[float, float]
             ],
             gradient: SignedMeasure[_Tspc],
             sigma_0: float,
             sigma_1: float,
             set_start: SimilarityClass[_Tspc],
             set_step: SimilarityClass[_Tspc],
             set_end: SimilarityClass[_Tspc] | None = None,
             *,
             err_bnd: float = math.inf,
             err_decay: float = 0.5
             ) -> tuple[tuple[float, float, float, float, float], float]:
    '''
    Evaluate step quality.

    Arguments
    ---------
    eval_func : (SimilarityClass, float) -> (float, float)
        Controlled evaluator for the objective functional. Receives a similarity
        class and an error bound and returns an evaluate and an error bound.

    gradient : SignedMeasure
        Evaluated gradient. Must assign strictly negative measure to the step
        set.

    sigma_0 : float
        Acceptance threshold. Must be strictly between `0` and `1`.

    sigma_1 : float
        Rejection threshold. Must be strictly between `sigma_0` and `1`.

    set_start : SimilarityClass
        Start point of the step.

    set_step : SimilarityClass
        Step.

    set_end : SimilarityClass (optional)
        End point of the step. If `None`, it is calculated from `set_start` and
        `set_step`. Pre-calculation is recommended to avoid multiple evaluation.
        Defaults to `None`.

    err_bnd : float (optional, keyword-only)
        Initial error bound for the controlled evaluation loop. Must be strictly
        greater than zero. Can be infinite. Defaults to positive infinity.

    err_decay : float (optional, keyword-only)
        Error decay rate for the controlled evaluation loop. Must be strictly
        between `0` and `1`. Defaults to `0.5`.

    Returns
    -------
    (f1, e1, f2, e2, rho) : tuple of 5 floats
    
        f1 : float
            Functional value at start point.

        e1 : float
            Functional error at start point.

        f2 : float
            Functional value at end point.

        e2 : float
            Functional error at end point.

        rho : float
            Approximate step quality.

    e_rho : float
        Error estimate between approximate and semi-approximate step quality.
        Always no greater than the maximum of `rho - sigma_0` and
        `sigma_1 - rho`.
    '''
    # Fill in endpoint if not provided.
    if set_end is None:
        set_end = set_start ^ set_step
        if isinstance(set_end, NotImplementedType):
            raise TypeError('Symmetric difference not implemented.')

    # Project change in objective
    proj_chg = gradient(set_step)

    # Define evaluator
    def evaluator(
        err_bnd: float
    ) -> tuple[
        tuple[
            float, float,   # Start value and error
            float, float,   # End value and error
            float           # Step quality
        ],
        float               # Step quality error
    ]:
        f1, e1 = eval_func(set_start, err_bnd / 2 * abs(proj_chg))
        f2, e2 = eval_func(set_end, err_bnd / 2 * abs(proj_chg))
        return (
            f1, e1,
            f2, e2,
            (f2 - f1) / proj_chg
        ), (e1 + e2) / abs(proj_chg)

    # Define bound update function
    def bound_update(
        value: tuple[float, float, float, float, float]
    ) -> float:
        *_, rho = value
        return max(rho - sigma_0, sigma_1 - rho)

    return controlled_eval(evaluator, bound_update, err_bnd=err_bnd,
                           err_decay=err_decay)


def eval_grad(
    grad: Callable[
        [SimilarityClass[_Tspc], float],
        tuple[SignedMeasure[_Tspc], float]
    ],
    set_cur: SimilarityClass[_Tspc],
    instat_tol: float,
    tr_rad: float,
    xi_tau: float,
    xi_delta: float,
    xi_g: float,
    step_qual: float,
    err_norm: ErrorNorm,
    *,
    err_bnd: float,
    err_decay: float,
    instat_eval: Callable[
        [SignedMeasure[_Tspc]],
        tuple[SimilarityClass[_Tspc], float]
    ] | None = None
) -> tuple[SignedMeasure[_Tspc], SimilarityClass[_Tspc], float, float]:
    '''
    Evaluate gradient.

    Argument
    --------
    grad : (SimilarityClass[T], float) -> (float, float)
        Controlled gradient evaluator.

    set_cur : SimilarityClass[T]
        Solution at which the gradient should be evaluated.

    instat_tol : float
        Instationarity tolerance. Must be strictly positive.

    tr_rad : float
        Trust region radius. Must be strictly positive and no greater than
        the measure of the universal set.

    xi_tau : float
        Relative error tolerance for instationarity test. Must be strictly
        between `0` and `1`.

    xi_delta : float
        Relative error tolerance for step determination. Must be strictly
        between `0` and `1`.

    xi_g : float
        Relative error tolerance for projected change. Must be strictly between
        `0` and `1`.

    step_qual : float
        Quality constant for unconstrained step determination method. Must be
        strictly greater than `0`.

    err_norm : ErrorNorm
        Norm according to which gradient errors are controlled.

    err_bnd : float (keyword-only)
        Initial error bound for the controlled evaluation loop. Must be
        strictly greater than `0`. Can be infinite.

    err_decay : float (keyword-only)
        Decay rate for the controlled evaluation loop. Must be strictly
        between `0` and `1`.

    instat_eval : SignedMeasure[S] -> float (optional, keyword-only)
        Optional instationarity evaluator. Can be used to carry cached
        instationarity to the caller.

    Returns
    -------
    g : SignedMeasure[T]
        Evaluate of the gradient. Instationarity error is guaranteed to be
        no greater than the product of `xi_tau` and the maximum of
        `instat_tol` and the difference between approximate instationarity
        and `instat_tol`. Projected change error is guaranteed to be no
        greater than the product of `xi_g` and the absolute value of the
        approximate projected change.

    set_neg : SimilarityClass[T]
        Strict sublevel set for `g` and level `0`.

    tau : float
        Approximate instationarity.

    e : float
        Upper bound on the gradient error with respect to the control norm.
    '''
    # Set up instationarity evaluator
    if instat_eval is None:
        instat_eval = eval_instat(set_cur.space)

    # Inner evaluator
    def phi(beta: float) -> tuple[tuple[SignedMeasure[_Tspc], SimilarityClass[_Tspc], float, float], float]:
        g, e = grad(set_cur, beta)
        set_neg, tau = instat_eval(g)
        return (g, set_neg, tau, e), e

    # Bound oracle
    def omega(val: tuple[SignedMeasure[_Tspc], SimilarityClass[_Tspc], float, float]) -> float:
        g, set_neg, tau, e = val
        omega_tau = xi_tau * max(instat_tol, tau - instat_tol)
        proj_chg_bnd = (1 - xi_delta) * step_qual * (tr_rad / max(tr_rad, set_neg.measure)) * max(tau, instat_tol)
        omega_d = xi_g * proj_chg_bnd

        if err_norm is ErrorNorm.L1:
            return min(omega_tau, omega_d)
        elif err_norm is ErrorNorm.Linfty:
            meas = (g < e).measure
            if meas > 0:
                return min(omega_tau / (g < e).measure, omega_d / tr_rad)
            else:
                return omega_d / tr_rad
        else:
            assert_never(err_norm)

    # Invoke controlled_eval
    v, _ = controlled_eval(phi, omega, err_bnd=err_bnd, err_decay=err_decay)
    return v
