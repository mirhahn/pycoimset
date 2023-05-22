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
"""
Implementations of the basic unconstrained optimization loop.
"""

from dataclasses import dataclass
import math
from typing import IO, Generic, Optional, TypeVar, cast
from warnings import warn

from ..problem import Problem
from ..step import SteepestDescentStepFinder
from ..typing import (
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)

__all__ = ['UnconstrainedSolver']


Spc = TypeVar('Spc', bound=SimilaritySpace)
OtherSpc = TypeVar('OtherSpc', bound=SimilaritySpace)


class UnconstrainedSolver(Generic[Spc]):
    """
    Unconstrained optimization loop.

    This is a barebones implementation of the controlled descent framework
    presented in Section 3.1 of the thesis.
    """

    @dataclass
    class Parameters:
        '''
        User specified parameters for the algorithm.
        '''

        #: Absolute stationarity tolerance.
        abstol: float = 1e-3

        #: Trust region reduction threshold. Must be strictly between 0 and 1.
        sigma_low: float = 0.1

        #: Trust region enlargement threshold. Must be strictly between
        #: `sigma_low` and 1.
        sigma_high: float = 0.9

        #: First error tuning parameter. Must be strictly between 0 and
        #: :code:`1 - sigma_low`.
        zeta_1: float = 0.5

        #: Second error tuning parameter. Must be strictly between 0 and
        #: 0.5.
        zeta_2: float = 0.01

        #: Initial trust region radius. This will be clamped to be a number
        #: strictly greater than 0 and less than or equal to the maximal step
        #: size possible in the variable space upon initialization.
        tr_radius: float = math.inf

        #: Maximum number of iterations.
        max_iter: Optional[int] = None

    @dataclass
    class State(Generic[OtherSpc]):
        '''
        Internal state maintained during iteration.
        '''

        #: Variable values.
        x: SimilarityClass[OtherSpc]

        #: Objective function value during last evaluation.
        f: Optional[float] = None

        #: Objective function error bound during last evaluation.
        e_f: Optional[float] = None

        #: Gradient measures obtained during last evaluation.
        g: Optional[SignedMeasure] = None

        #: Gradient error obtained during last evaluation. The precise meaning
        #: of this number differs based on whether the objective functional is
        #: :math:`L^1` or :math:`L^\infty` controlled.
        e_g: Optional[float] = None

        #: Instationarity measure during last evaluation.
        dual_inf: Optional[float] = None

        #: Error of instationarity measure during last evaluation.
        err_dual_inf: Optional[float] = None

        #: Current trust region radius.
        tr_radius: Optional[float] = None

    @dataclass
    class Stats:
        '''
        Statistics collected during the optimization loop.

        This is useful for post-optimization performance evaluation. The
        structure can be re-used in subsequent runs. In this case, statistics
        accumulate.
        '''

        #: Total number of iterations including both accepted and rejected
        #: steps.
        n_iter: int = 0

        #: Number of objective function evaluations.
        n_fun_eval: int = 0

        #: Number of gradient evaluations.
        n_grad_eval: int = 0

        #: Number of times that a step was accepted.
        n_steps_accepted: int = 0

        #: Number of times that a step was rejected.
        n_steps_rejected: int = 0

        #: Total wall time spent in the optimization loop (in seconds).
        t_total: float = 0.0

        #: Total wall time spent evaluating the objective (in seconds).
        t_fun_eval: float = 0.0

        #: Total wall time spent evaluating the gradient (in seconds).
        t_grad_eval: float = 0.0

    _prob: Problem[Spc]
    _param: Parameters
    _state: State[Spc]
    _stats: Stats
    _step: UnconstrainedStepFinder[Spc]

    def __init__(self, problem: Problem[Spc],
                 step_finder: Optional[UnconstrainedStepFinder[Spc]] = None):
        if len(problem.constraints) > 0:
            raise ValueError('Cannot solve problem with constraints.')

        # Store a reference to the problem.
        self._prob = problem

        # Set up parameter structure.
        self._param = UnconstrainedSolver.Parameters()

        # Set up the internal state.
        self._state = UnconstrainedSolver.State(x=problem.initial_value)

        # Set up stats.
        self._stats = UnconstrainedSolver.Stats()

        # Set up step finder.
        if step_finder is None:
            self._step = SteepestDescentStepFinder[Spc]()
        else:
            self._step = step_finder

    @property
    def param(self) -> Parameters:
        '''Algorithmic parameters.'''
        return self._param

    @property
    def solution(self) -> SimilarityClass:
        '''Current solution.'''
        return self._state.x

    def _logline(self, iter: int, objval: float, instat: float,
                 step: Optional[float] = None, tr_fail: Optional[int] = None,
                 file: Optional[IO] = None, flush: bool = False):
        '''
        Output a single log line indicating the status of the solver.
        '''
        # Print header
        if iter % 50 == 0:
            print('iter |      obj      |     instat    |      step     | '
                  'tr_fail', file=file)
        print(f'{iter:4d} | {objval:13.6e} | {instat:13.6e} | ', end='',
              file=file)
        if step is None:
            print('          --- | ', end='', file=file)
        else:
            print(f'{step:13.6e} | ', end='', file=file)
        if tr_fail is None:
            print('    ---', end='', file=file)
        else:
            print(f'{tr_fail:7d}', end='', file=file)
        print(flush=flush, file=file)

    def _l1errbnd(self, tr_radius: float) -> float:
        '''
        Gradient error bound for :math:`L^1` controlled functional.

        This is an internal function. You should not call it directly.
        '''
        return self._param.abstol * min(
            self._param.zeta_2,
            (
                self._param.zeta_1
                * (1 - 2 * self._param.zeta_2)
                * self._step.quality
            ) / (
                self._prob.space.measure
                * (1 + self._param.zeta_1)
            ) * tr_radius
        )

    def _l1graderr_global(self, err_bnd: float) -> float:
        '''
        Gradient error bound valid for all product similarity classes.

        This is an internal function. You should not call it directly. This
        variant is intended for use with :math:`L^1` controlled functionals.
        '''
        return err_bnd

    def _l1graderr_local(self, err_bnd: float, _: SimilarityClass) -> float:
        '''
        Gradient error bound valid for a specific product similarity class.

        This is an internal function. You should not call it directly. This
        variant is intended for use with :math:`L^1` controlled functionals.
        '''
        return err_bnd

    def _linferrbnd(self, _: float) -> float:
        '''
        Gradient error bound for :math:`L^\\infty` controlled functional.

        This is an internal function. You should not call it directly.
        '''
        max_step = self._prob.space.measure
        return self._param.abstol * min(
            self._param.zeta_2 / max_step,
            (
                self._param.zeta_1
                * (1 - 2 * self._param.zeta_2)
                * self._step.quality
            ) / (max_step * (1 + self._param.zeta_1))
        )

    def _linfgraderr_global(self, err_bnd: float) -> float:
        '''
        Gradient error bound valid for all product similarity classes.

        This is an internal function. You should not call it directly. This
        variant is intended for use with :math:`L^\\infty` controlled
        functionals.
        '''
        return err_bnd * self._prob.space.measure

    def _linfgraderr_local(self, err_bnd: float, cls: SimilarityClass
                           ) -> float:
        '''
        Gradient error bound valid for a specific product similarity class.

        This is an internal function. You should not call it directly. This
        variant is intended for use with :math:`L^\\infty` controlled
        functionals.
        '''
        return err_bnd * cls.measure

    def solve(self):
        '''
        Run the main optimization loop.
        '''
        # Introduce shorthands for the four data objects.
        problem = self._prob
        par = self._param
        state = self._state
        stats = self._stats
        step_finder = self._step

        # Sanitize parameters.
        if par.abstol <= 0.0:
            par.abstol = 1e-3
            warn('param.abstol must be strictly positive (reset to 1e-3)')
        if par.sigma_low <= 0 or par.sigma_low >= 1:
            par.sigma_low = 0.1
            warn('param.sigma_low must be strictly between 0 and 1 (reset to '
                 '0.1)')
        if par.sigma_high <= par.sigma_low:
            par.sigma_high = 2 * par.sigma_low
            warn('param.sigma_high must be greater than param.sigma_low ('
                 f'reset to {par.sigma_high})')
        if par.zeta_1 <= 0 or par.zeta_1 >= 1 - par.sigma_low:
            par.zeta_1 = 0.5 * (1 - par.sigma_low)
            warn('param.zeta_1 must be strictly between 0 and 1 - '
                 f'param.sigma_low (reset to {par.zeta_1})')
        if par.zeta_2 <= 0 or par.zeta_2 >= 0.5:
            par.zeta_2 = 0.1
            warn('param.zeta_2 must be strictly between 0 and 0.5 (reset to '
                 f'{par.zeta_2})')

        # Helper functions for error bounds.
        if problem.objective.grad_tol_type == 'l1':
            errbnd = self._l1errbnd
            graderr_global = self._l1graderr_global
            graderr_local = self._l1graderr_local
        elif problem.objective.grad_tol_type == 'linfty':
            errbnd = self._linferrbnd
            graderr_global = self._linfgraderr_global
            graderr_local = self._linfgraderr_local
        else:
            raise ValueError('Gradient tolerance type '
                             f'\'{problem.objective.grad_tol_type}\' is '
                             'unknown.')

        # Initialize trust region radius if necessary.
        if state.tr_radius is None:
            state.tr_radius = par.tr_radius
        if state.tr_radius <= 0.0:
            state.tr_radius = math.inf
        state.tr_radius = min(state.tr_radius, problem.space.measure)

        # Perform initial gradient evaluation if necessary.
        problem.objective.arg = state.x
        problem.objective.grad_tol = errbnd(state.tr_radius)
        state.g, state.e_g = problem.objective.get_gradient()
        state.f, state.e_f = problem.objective.get_value()

        # Calculate dual infeasibility.
        full_step = state.g < 0.0
        state.dual_inf = -state.g(full_step)
        state.err_dual_inf = graderr_global(state.e_g)

        # Output log line.
        self._logline(stats.n_iter, state.f, state.dual_inf)

        failed_tr_count = 0
        while (par.max_iter is None or stats.n_iter < par.max_iter):
            # Terminate if near-stationary.
            if state.dual_inf <= par.abstol - state.err_dual_inf:
                break

            # Find an improvement step.
            step_finder.gradient = state.g
            step_finder.tolerance = (
                (par.zeta_2 * step_finder.quality * par.abstol)
                / problem.space.measure
                * state.tr_radius
            )
            step_finder.radius = state.tr_radius
            step, _ = step_finder.get_step()

            # Calculate projected objective change and error bound.
            proj_diff = state.g(step)
            proj_err = graderr_local(state.e_g, step)

            # Calculate objective function evaluation error budget.
            error_budget = par.sigma_low * min(
                (abs(proj_diff) - proj_err) * par.zeta_1 - proj_err,
                (abs(proj_diff) * par.zeta_1 - proj_err) / (1 - par.zeta_1)
            )

            # Evaluate functional at current point if necessary.
            if state.f is None or state.e_f is None or \
                    state.e_f > (2 / 3) * error_budget:
                problem.objective.val_tol = (2/3) * error_budget
                state.f, state.e_f = problem.objective.get_value()

            # Evaluate functional at end point.
            problem.objective.arg = state.x ^ step
            problem.objective.val_tol = min((2/3) * error_budget,
                                            cast(float, state.e_f))
            f_end, e_f_end = problem.objective.get_value()

            # Decide whether to accept or reject the step.
            rho = (f_end - state.f) / proj_diff
            if rho >= par.sigma_low:
                # Update state
                state.x, state.f, state.e_f = problem.objective.arg, f_end, \
                    e_f_end

                # Update trust region radius
                if rho >= par.sigma_high:
                    state.tr_radius = min(2 * state.tr_radius,
                                          problem.space.measure)

                # Evaluate gradient
                problem.objective.grad_tol = errbnd(state.tr_radius)
                state.g, state.e_g = problem.objective.get_gradient()

                # Calculate dual infeasibility.
                full_step = state.g < 0.0
                state.dual_inf = -state.g(full_step)
                state.err_dual_inf = graderr_global(state.e_g)

                # Output log line.
                self._logline(stats.n_iter + 1, state.f, state.dual_inf,
                              step.measure, failed_tr_count)

                # Update stats.
                stats.n_steps_accepted += 1

                # Reset failed trust-region count.
                failed_tr_count = 0
            else:
                # Update trust region radius.
                state.tr_radius /= 2

                # Reset argument of objective.
                problem.objective.arg = state.x

                # Re-evaluate gradient if necessary.
                problem.objective.grad_tol = errbnd(state.tr_radius)
                state.g, state.e_g = problem.objective.get_gradient()

                # Calculate dual infeasibility.
                full_step = state.g < 0.0
                state.dual_inf = -state.g(full_step)
                state.err_dual_inf = graderr_global(state.e_g)

                # Update stats.
                stats.n_steps_rejected += 1

                # Increase failed trust-region count.
                failed_tr_count += 1
                print(f'DEBUG: rejection {failed_tr_count}', flush=True)
                print(f'DEBUG: rho = {rho}', flush=True)
                print(f'DEBUG: tr_radius = {2 * state.tr_radius}', flush=True)
                print(f'DEBUG: step_size = {step.measure}', flush=True)

            # Advance the iteration counter.
            stats.n_iter += 1
