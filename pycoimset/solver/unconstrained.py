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
"""
Implementation of the basic unconstrained optimization loop.
"""

from dataclasses import dataclass
from enum import IntEnum
import logging
import math
import time
from typing import Callable, Generic, NamedTuple, Optional, Self, TypeVar

import numpy

from ..logging import TabularLogger
from ..step import SteepestDescentStepFinder
from ..typing import (
    ErrorNorm,
    Functional,
    JSONSerializable,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)

__all__ = ['Solver']


Spc = TypeVar('Spc', bound=SimilaritySpace)
T = TypeVar('T')


logger = logging.getLogger(__name__)


@dataclass
class SolverParameters(JSONSerializable):
    '''
    User specified parameters for the algorithm.

    Attributes
    ----------
    abstol : float
        Absolute instationarity tolerance. Reference value for the
        termination criterion. Defaults to ``1e-3``.
    thres_accept : float
        Model quality threshold above which steps are accepted. Defaults
        to ``0.2``.
    thres_reject : float
        Model quality threshold below which steps are rejected. Must be
        strictly between `thres_accept` and ``1.0``.
    thres_tr_expand : float
        Model quality threshold above which the trust region is expanded.
        Must be strictly greater than `thres_accept`.
    margin_step : float
        Error tuning parameter. Bounds the ratio between the step's
        projected descent and the maximal projected descent for any
        step within the trust region. Must be strictly between
        ``0.0`` and ``1.0``. Defaults to ``0.25``.
    margin_proj_desc : float
        Error tuning parameter. Bounds the admissible relative error of
        the projected step for the given step. Must be strictly between
        ``0.0`` and ``1.0``. Defaults to ``0.1``.
    margin_instat : float
        Error tuning parameter. Bounds the ratio between instationarity
        error and absolute termination tolerance. Must be strictly
        between ``0.0`` and ``1.0``. Defaults to ``0.5``.
    tr_radius : float, optional
        Initial trust region radius. Clamped to search space diameter.
        Defaults to `None`.
    max_iter : int, optional
        Iteration limit. Defaults to `None`.
    '''

    #: Absolute stationarity tolerance.
    abstol: float = 1e-3

    #: Acceptance threshold. Must be in (0, 1).
    thres_accept: float = 0.2

    #: Trust region reduction threshold. Must be strictly between
    #: `thres_accept` and 1.
    thres_reject: float = 0.4

    #: Trust region enlargement threshold. Must be strictly between
    #: `thres_accept` and 1.
    thres_tr_expand: float = 0.6

    #: Error tuning parameter. Regulates the ratio between a step's
    #: projected descent and the maximum guaranteeable projected
    #: descent. Must be strictly between 0 and 1.
    margin_step: float = 0.25

    #: Error tuning parameter. Regulates the relative error of the
    #: projected descent for the given step. Must be strictly between
    #: 0 and `1 - thres_reject`.
    margin_proj_desc: float = 0.1

    #: Error tuning parameter. Regulates the ratio between
    #: instationarity error and absolute termination tolerance.
    #: Must be strictly between 0 and 1.
    margin_instat: float = 0.5

    #: Initial trust region radius. This will be clamped to be a number
    #: strictly greater than 0 and less than or equal to the maximal step
    #: size possible in the variable space upon initialization.
    tr_radius: Optional[float] = None

    #: Maximum number of iterations.
    max_iter: Optional[int] = None

    def sanitize(self) -> None:
        '''
        Sanitizes parameters.
        '''
        if self.max_iter is not None and self.max_iter < 0:
            self.max_iter = None

        if self.tr_radius is not None and self.tr_radius <= 0.0:
            self.tr_radius = None

        if self.margin_step <= 0.0 or self.margin_step >= 1.0:
            self.margin_step = 0.5

        if self.margin_proj_desc <= 0.0 or self.margin_proj_desc >= 1.0:
            self.margin_proj_desc = 0.1

        if self.margin_instat <= 0.0 or self.margin_instat >= 1.0:
            self.margin_instat = 0.5

        if self.thres_reject <= 0.0 or self.thres_reject >= 1.0:
            self.thres_reject = 0.4

        if (self.thres_accept <= 0.0
                or self.thres_accept >= self.thres_reject):
            self.thres_accept = self.thres_reject / 2

        if (self.thres_tr_expand <= self.thres_accept
                or self.thres_accept >= 1.0):
            self.thes_tr_expand = (self.thres_accept + 1.0) / 2

        if self.abstol <= 0.0:
            self.abstol = 1e-3


@dataclass(slots=True)
class SolverStats:
    '''
    Statistics collected during the optimization loop.

    This is useful for post-optimization performance evaluation. The
    structure can be re-used in subsequent runs.
    '''

    #: Total number of iterations including both accepted and rejected
    #: steps.
    n_iter: int = 0

    #: Total wall time spent in the optimization loop (in seconds).
    t_total: float = 0.0

    #: Measure of last step.
    last_step: float = 0.0

    #: Number of rejected steps.
    n_reject: int = 0


class SolverStatus(IntEnum):
    Running = 0
    Solved = 1
    UnknownError = 64,
    UserInterruption = 65
    SmallStep = 66
    IterationMaximum = 67

    Message: dict[Self, str]

    @property
    def is_running(self) -> bool:
        return self == SolverStatus.Running

    @property
    def is_error(self) -> bool:
        return self >= SolverStatus.UnknownError
    
    @property
    def message(self) -> str:
        '''Describe status.'''
        return type(self).Message.get(self, '(missing status message)')


SolverStatus.Message = {
    SolverStatus.Running: "still running",
    SolverStatus.Solved: "solution found",
    SolverStatus.UnknownError: "unknown error",
    SolverStatus.IterationMaximum: "iteration maximum exceeded",
    SolverStatus.SmallStep: "step too small",
    SolverStatus.UserInterruption: "interrupted by user",
}


class ValueErrorPair(NamedTuple, Generic[T]):
    value: T
    error: float


class Solution(Generic[Spc]):
    '''
    Representation of a solution to an optimization problem.

    A solution consists of a point in search space (i.e., a similarity
    class) alongside potential evaluation results for objective
    function and gradient.
    '''
    _func: Functional[Spc]
    _arg: SimilarityClass[Spc]
    _valtol: float
    _gradtol: float
    _val: Optional[ValueErrorPair[float]]
    _grad: Optional[ValueErrorPair[SignedMeasure[Spc]]]
    _fullstep: Optional[SimilarityClass[Spc]]
    _instat: Optional[ValueErrorPair[float]]

    def __init__(self, func: Functional[Spc], arg: SimilarityClass[Spc]):
        self._func = func
        self._arg = arg
        self._valtol = math.inf
        self._gradtol = math.inf
        self._val = None
        self._grad = None
        self._fullstep = None
        self._instat = None

    @property
    def arg(self) -> SimilarityClass[Spc]:
        '''Argument for the objective functional.'''
        return self._arg

    @property
    def val_tol(self) -> float:
        '''Error tolerance for objective values.'''
        return self._valtol

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        if tol <= 0.0:
            raise ValueError('tolerance must be strictly positive')
        self._valtol = tol
        if self._val is not None and self._val.error > tol:
            self._val = None

    @property
    def grad_tol(self) -> float:
        '''Error tolerance for gradients.'''
        return self._gradtol

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        if tol <= 0.0:
            raise ValueError('tolerance must be strictly positive')
        self._gradtol = tol
        if self._grad is not None and self._grad.error > tol:
            self._grad = None
            self._fullstep = None
            self._instat = None

    @property
    def val(self) -> ValueErrorPair[float]:
        '''Objective function value.'''
        if self._val is not None:
            return self._val

        self._func.arg = self._arg
        self._func.val_tol = self._valtol
        self._func.grad_tol = self._gradtol
        self._val = ValueErrorPair[float](*self._func.get_value())

        return self._val

    @property
    def grad(self) -> ValueErrorPair[SignedMeasure[Spc]]:
        '''Gradient value.'''
        if self._grad is not None:
            return self._grad

        self._func.arg = self._arg
        self._func.val_tol = self._valtol
        self._func.grad_tol = self._gradtol
        self._grad = ValueErrorPair[SignedMeasure[Spc]](
            *self._func.get_gradient()
        )

        return self._grad

    @property
    def full_step(self) -> SimilarityClass[Spc]:
        '''Full step.'''
        if self._fullstep is not None:
            return self._fullstep

        grad = self.grad
        self._fullstep = grad.value < 0.0

        return self._fullstep

    @property
    def instationarity(self) -> ValueErrorPair[float]:
        '''Approximate instationarity.'''
        if self._instat is not None:
            return self._instat

        grad = self.grad
        step = self.full_step
        norm = self._func.grad_tol_type
        self._instat = ValueErrorPair[float](
            -grad.value(step),
            norm.estimated_error(step.measure, grad.error)
        )

        return self._instat

    def invalidate(self) -> None:
        '''Invalidate all cached results.'''
        self._instat = None
        self._fullstep = None
        self._grad = None
        self._val = None


class Solver(Generic[Spc]):
    """
    Unconstrained optimization loop.

    This is a barebones implementation of the controlled descent framework
    presented in Section 3.1 of the thesis.
    """
    #: Problem description.
    objective: Functional[Spc]

    #: Step finder.
    step_finder: UnconstrainedStepFinder[Spc]

    #: Current status flag.
    status: SolverStatus

    #: Parameters.
    param: SolverParameters

    #: Statistics.
    stats: SolverStats

    #: Current trust region radius.
    radius: float

    #: Current solution.
    solution: Solution[Spc]

    #: Estimate of objective curvature.
    curvature: float

    #: Callback.
    callback: Optional[Callable[[Self], None]]

    #: Logger for tabular output.
    logger: TabularLogger

    def __init__(self, obj_func: Functional[Spc],
                 step_finder: Optional[UnconstrainedStepFinder[Spc]] = None,
                 initial_sol: Optional[SimilarityClass[Spc]] = None,
                 param: Optional[SolverParameters] = None,
                 callback: Optional[Callable[[Self], None]] = None,
                 **kwargs):
        # Retain reference to objective.
        self.objective = obj_func

        # Set up step finder.
        if step_finder is None:
            self.step_finder = SteepestDescentStepFinder[Spc]()
        else:
            self.step_finder = step_finder

        # Set up solution.
        if initial_sol is None:
            initial_sol = obj_func.input_space.empty_class
        self.solution = Solution[Spc](self.objective, initial_sol)

        # Set up parameter structure.
        if param is not None:
            self.param = param
        else:
            self.param = SolverParameters(**kwargs)

        # Set up remaining data.
        self.status = SolverStatus.Running
        self.radius = max(0.0, min(self.param.tr_radius
                                   if self.param.tr_radius is not None
                                   else math.inf,
                                   obj_func.input_space.measure))
        self.curvature = 0.0
        self.stats = SolverStats()

        # Sanitize parameters.
        self.param.sanitize()

        # Set up callback.
        self.callback = callback

        # Set up logger.
        self.logger = TabularLogger(
            cols=['time', 'iter', 'obj', 'instat', 'step', 'tr_fail'],
            format={
                'time': '8.2f',
                'iter': '4d',
                'obj': '13.6e',
                'instat': '13.6e',
                'step': '13.6e',
                'tr_fail': '7d'
            },
            width={
                'time': 8,
                'iter': 4,
                'obj': 13,
                'instat': 13,
                'step': 13,
                'tr_fail': 7
            },
            flush=True
        )

    def _tolerances(self,
                    tau: Optional[float] = None,
                    rho: Optional[float] = None
                    ) -> tuple[float, float, float]:
        '''
        Choose appropriate error tolerances.

        This method is used by the optimization loop to determine
        evaluation error margins for the objective functional. It uses
        prior guesses for instationarity and step quality.

        Parameters
        ----------
        tau : float, optional
            Anticipated unsigned instationarity.
        rho : float, optional
            Anticipated step quality.

        Returns
        -------
        tuple[float, float, float]
            Objective error tolerance, gradient error tolerance, and step
            error tolerance in that order.
        '''
        # Get parameters.
        eps = self.param.abstol
        xi_step = self.param.margin_step
        xi_prdesc = self.param.margin_proj_desc
        xi_instat = self.param.margin_instat
        step_quality = self.step_finder.quality
        radius = self.radius
        mu_full = self.solution.full_step.measure
        norm = self.objective.grad_tol_type

        # Guess instationarity if no guess is given.
        if tau is None:
            tau = max(eps, self.solution.instationarity.value)

        # Approximate upper bound to full step measure.
        mu_full_ub = self.objective.input_space.measure
        if norm is ErrorNorm.Linfty:
            grad = self.solution.grad
            mu_full_ub = (grad.value < grad.error).measure

        # Calculate lower bound for projected descent.
        expected_step_ratio = step_quality * min(
            1.0, radius / mu_full
        )
        proj_desc_min = (1.0 - xi_step) * expected_step_ratio * tau

        # Guess step quality if no guess is given
        if rho is None:
            rho = 1.0 - (
                self.curvature * min(radius, mu_full)**2 / (2 * proj_desc_min)
            )

        # Calculate gradient error tolerance.
        grad_tol_step = norm.required_tolerance(
            radius, xi_prdesc * proj_desc_min
        )
        grad_tol_full = norm.required_tolerance(
            mu_full_ub, xi_instat * max(tau - eps, eps)
        )
        grad_tol = min(grad_tol_step, grad_tol_full) / 2

        # Calculate objective error tolerance.
        sigma_0 = self.param.thres_accept
        sigma_1 = self.param.thres_reject
        rho_tol = max(rho - sigma_0, sigma_1 - rho, (sigma_1 - sigma_0) / 4)
        obj_tol = rho_tol * proj_desc_min / 2

        # Calculate step tolerance.
        step_tol = xi_step * expected_step_ratio * tau

        return obj_tol, grad_tol, step_tol

    def _update_curvature(self, step_size: float, projected_descent: float,
                          actual_descent: float) -> None:
        '''
        Update curvature estimate.

        Parameters
        ----------
        step_size : float
            Measure of the step.
        projected_descent : float
            Projected descent given the linear model function.
        actual_descent : float
            Actual descent associated with the step.
        '''
        curvature = 2 * (actual_descent - projected_descent) / step_size**2
        if math.isfinite(curvature) and curvature >= 0.0:
            self.curvature = 0.9 * self.curvature + 0.1 * curvature

    def step(self) -> None:
        '''
        Perform a single optimization step.
        '''
        # Get error norm.
        err_norm = self.objective.grad_tol_type

        # Evaluate instationarity.
        while True:
            obj_tol, grad_tol, step_tol = self._tolerances(tau=self.solution.instationarity.value)
            if self.solution.instationarity.error <= self.param.margin_instat * max(
                self.solution.instationarity.value - self.param.abstol, self.param.abstol
            ):
                break
            self.solution.val_tol = obj_tol
            self.solution.grad_tol = grad_tol

        # Perform stationarity test.
        if self.solution.instationarity.value <= self.param.abstol:
            self.status = SolverStatus.Solved
            return

        accepted = False
        while not accepted:
            # Find step.
            assert self.solution.grad.error <= self.solution.grad_tol
            self.step_finder.gradient = self.solution.grad.value
            self.step_finder.radius = self.radius
            self.step_finder.tolerance = step_tol
            step, _ = self.step_finder.get_step()

            # Find projected change.
            proj_chg = self.solution.grad.value(step)
            proj_chg_error = err_norm.estimated_error(
                step.measure, self.solution.grad.error
            )
            if proj_chg_error > self.param.margin_proj_desc * abs(proj_chg):
                logger.info(f'projected change error {proj_chg_error} > {self.param.margin_proj_desc * abs(proj_chg)}')
                obj_tol, grad_tol, step_tol = self._tolerances(self.solution.instationarity.value, proj_chg / step.measure)
                self.solution.val_tol = obj_tol
                self.solution.grad_tol = grad_tol
                continue

            # Calculate step quality.
            new_sol = Solution(self.objective, self.solution.arg ^ step)
            new_sol.val_tol = obj_tol
            new_sol.grad_tol = grad_tol
            step_quality = ((new_sol.val.value - self.solution.val.value)
                / proj_chg)
            step_quality_err = ((self.solution.val.error + new_sol.val.error)
                / abs(proj_chg))

            # Update curvature estimate.
            self._update_curvature(step.measure, proj_chg,
                                   new_sol.val.value - self.solution.val.value)

            # Assess whether or not to accept.
            if step_quality - step_quality_err >= self.param.thres_accept:
                # Accept new solution.
                self.solution = new_sol

                # Increment iteration counter and output log line.
                self.stats.n_iter += 1
                self.stats.last_step = step.measure

                # Check whether to increase the trust region radius.
                if step_quality >= self.param.thres_tr_expand:
                    self.radius = min(
                        self.objective.input_space.measure, 2 * self.radius
                    )

                accepted = True
            elif step_quality + step_quality_err < self.param.thres_reject:
                # Log rejection.
                logger.info(f'rejecting step with rho = {step_quality}')

                # Decrease trust region radius.
                self.radius /= 2

                if self.radius < 1000 * numpy.finfo(float).eps:
                    self.status = SolverStatus.SmallStep
                    return

                # Increase rejection counter.
                self.stats.n_reject += 1
            else:
                # Log insufficient precision event.
                logger.info(f'step acceptance test inconclusive with rho = {step_quality} +- {step_quality_err}')

            # Adjust error tolerances.
            obj_tol, grad_tol, step_tol = self._tolerances(self.solution.instationarity.value, step_quality)
            self.solution.val_tol = obj_tol
            self.solution.grad_tol = grad_tol

    def solve(self) -> None:
        '''
        Run the main optimization loop.
        '''
        # Record start time.
        start_time = time.perf_counter()
        
        # Print initial log line.
        self.logger.push_line(
            time=time.perf_counter() - start_time,
            iter=self.stats.n_iter,
            obj=self.solution.val.value,
            instat=self.solution.instationarity.value
        )
        if self.callback is not None:
            self.callback(self)

        old_iter = self.stats.n_iter
        max_iter = self.param.max_iter
        self.status = SolverStatus.Running
        while (self.status.is_running and
               (max_iter is None or self.stats.n_iter - old_iter < max_iter)):
            old_reject = self.stats.n_reject
            self.step()
            self.logger.push_line(
                time=time.perf_counter() - start_time,
                iter=self.stats.n_iter,
                obj=self.solution.val.value,
                instat=self.solution.instationarity.value,
                step=self.stats.last_step,
                tr_fail=self.stats.n_reject - old_reject
            )
            if self.callback is not None:
                self.callback(self)

        if self.status.is_running:
            self.status = SolverStatus.IterationMaximum

        # Print termination reason.
        print(f'Terminated: {self.status.message}')
