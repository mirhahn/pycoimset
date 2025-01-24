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
"""
Implementation of the basic unconstrained optimization loop.
"""

from dataclasses import dataclass
from enum import IntEnum
import logging
import math
import signal
import time
from types import NotImplementedType
from typing import Callable, Generic, NamedTuple, Optional, Self, TypeVar

import numpy

from ...logging import TabularLogger
from ...step import SteepestDescentStepFinder
from ...typing import (
    Functional,
    JSONSerializable,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)
from ...util.signals import InterruptionFlag, interruptible_method
from .eval import eval_grad, eval_rho, make_func_eval, make_grad_eval

__all__ = ['UnconstrainedSolver']


Spc = TypeVar('Spc', bound=SimilaritySpace)
T = TypeVar('T')


logger = logging.getLogger(__name__)

# Find timing function
if hasattr(time, 'process_time'):
    get_time = time.process_time
elif hasattr(time, 'perf_counter'):
    get_time = time.perf_counter
else:
    get_time = time.monotonic


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

    #: Last objective value.
    last_obj_val: float = 0.0

    #: Last instationarity
    last_instat: float = 0.0

    #: Number of rejected steps.
    n_reject: int = 0


class SolverStatus(IntEnum):
    Running = 0
    Solved = 1
    UnknownError = 64
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


class UnconstrainedSolver(Generic[Spc]):
    """
    Unconstrained optimization loop.

    This is a barebones implementation of the controlled descent framework
    presented in Section 3.1 of the thesis.
    """
    #: Objective functional.
    f: Functional[Spc]

    #: Current iterate.
    x: SimilarityClass[Spc]

    #: Parameters.
    _p: SolverParameters

    #: Step finder.
    _step: UnconstrainedStepFinder[Spc]

    #: Current status flag.
    _status: SolverStatus

    #: Statistics.
    _stats: SolverStats

    #: Current trust region radius.
    _trrad: float

    #: Callback.
    callback: Optional[Callable[[Self], None]]

    #: Logger for tabular output.
    _log: TabularLogger

    def __init__(self, obj_func: Functional[Spc],
                 step_finder: Optional[UnconstrainedStepFinder[Spc]] = None,
                 initial_sol: Optional[SimilarityClass[Spc]] = None,
                 param: Optional[SolverParameters] = None,
                 callback: Optional[Callable[[Self], None]] = None,
                 **kwargs):
        # Set objective.
        self.f = obj_func

        # Set up step finder.
        self._step = step_finder if step_finder is not None else \
            SteepestDescentStepFinder[Spc]()

        # Set up solution.
        self.x = initial_sol if initial_sol is not None else \
            self.f.input_space.empty_class

        # Set up parameter structure.
        self._p = param if param is not None else SolverParameters(**kwargs)

        # Set up remaining data.
        self._status = SolverStatus.Running
        self._trrad = max(0.0, min(self._p.tr_radius
                                   if self._p.tr_radius is not None
                                   else math.inf,
                                   obj_func.input_space.measure))
        self._stats = SolverStats()

        # Sanitize parameters.
        self._p.sanitize()

        # Set up callback.
        self.callback = callback

        # Set up logger.
        self._log = TabularLogger(
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

    @property
    def stats(self) -> SolverStats:
        '''
        Statistics about the run.
        '''
        return self._stats

    @property
    def status(self) -> SolverStatus:
        '''
        Statistics about the run.
        '''
        return self._status

    @property
    def param(self) -> SolverParameters:
        '''
        Solver parameters.
        '''
        return self._p

    def _callback(self) -> float:
        '''
        Run callback and return runtime.
        '''
        if (cb := self.callback) is not None:
            start_time = get_time()
            cb(self)
            return get_time() - start_time
        return 0.0

    @interruptible_method(signal.SIGINT | signal.SIGTERM)
    def solve(self, int_flag: InterruptionFlag) -> None:
        '''
        Run the main optimization loop.
        '''
        # Simplified evaluators.
        func_val = make_func_eval(self.f, cache_size=2)
        grad_val = make_grad_eval(self.f, cache_size=2)

        # Record start time.
        cb_time = 0.0
        start_time = get_time()

        # Evaluate initial gradient.
        g, set_neg, instat, _ = eval_grad(grad_val,
                                          set_cur := self.x,
                                          eps_tau := self._p.abstol,
                                          tr_rad := self._trrad,
                                          xi_tau := self._p.margin_instat,
                                          xi_delta := self._p.margin_step,
                                          xi_g := self._p.margin_proj_desc,
                                          step_qual := self._step.quality,
                                          err_norm := self.f.grad_tol_type,
                                          err_bnd=math.inf,
                                          err_decay=0.9)

        # Print initial log line.
        self._log.push_line(
            time=get_time() - start_time - cb_time,
            iter=self._stats.n_iter,
            obj=(obj_val := func_val(self.x, math.inf)[0]),
            instat=instat
        )
        self._stats.last_obj_val = obj_val
        self._stats.last_instat = instat
        cb_time += self._callback()

        # Main optimization loop
        old_iter = self._stats.n_iter
        old_reject = self._stats.n_reject
        max_iter = self._p.max_iter
        self._status = SolverStatus.Running
        while (
            instat > eps_tau
            and int_flag.deferred_signal is None
            and (max_iter is None
                 or self._stats.n_iter - old_iter < max_iter)
        ):
            # Find step.
            self._step.gradient = g
            self._step.radius = tr_rad
            self._step.tolerance = xi_delta * step_qual * \
                (tr_rad / max(tr_rad, set_neg.measure)) * instat
            set_step, _ = self._step.get_step()
            set_next = set_cur ^ set_step

            if isinstance(set_next, NotImplementedType):
                raise NotImplementedError(
                    'symmetric difference not implemented'
                )

            # Find projected change.
            (_, _, obj_next, _, rho), e_rho = eval_rho(
                func_val,
                g,
                sigma_0 := self._p.thres_accept,
                sigma_1 := self._p.thres_reject,
                set_cur,
                set_step,
                set_next,
                err_bnd=math.inf,
                err_decay=0.9
            )

            # Assess whether or not to accept.
            if rho - sigma_0 >= e_rho:
                # Accept new solution.
                self.x = set_cur = set_next
                obj_val = obj_next

                # Increment iteration counter and output log line.
                self._stats.n_iter += 1
                self._stats.last_step = set_step.measure

                step_accepted = True

                # Check whether to increase the trust region radius.
                if rho >= self._p.thres_tr_expand:
                    self._trrad = tr_rad = min(
                        self.f.input_space.measure, 2 * tr_rad
                    )
            elif sigma_1 - rho > e_rho:
                # Log rejection.
                logger.info(f'rejecting step with rho = {rho}')

                # Decrease trust region radius.
                self._trrad = tr_rad = tr_rad / 2
                if self._trrad < 1000 * numpy.finfo(float).eps:
                    self._status = SolverStatus.SmallStep
                    return

                # Increase rejection counter.
                self._stats.n_reject += 1

                step_accepted = False
            else:
                raise RuntimeError('Step was neither accepted nor rejected. '
                                   'This is a bug and should never happen!')

            # Update gradient.
            g, set_neg, instat, _ = eval_grad(grad_val,
                                              set_cur,
                                              eps_tau,
                                              tr_rad,
                                              xi_tau,
                                              xi_delta,
                                              xi_g,
                                              step_qual,
                                              err_norm,
                                              err_bnd=math.inf,
                                              err_decay=0.9)

            # Print log line.
            if step_accepted:
                self._log.push_line(
                    time=get_time() - start_time - cb_time,
                    iter=self._stats.n_iter,
                    obj=obj_val,
                    instat=instat,
                    step=self._stats.last_step,
                    tr_fail=self._stats.n_reject - old_reject
                )
                self._stats.last_obj_val = obj_val
                self._stats.last_instat = instat
                old_reject = self._stats.n_reject
                cb_time += self._callback()

        if int_flag.deferred_signal is not None:
            self._status = SolverStatus.UserInterruption
        elif instat <= eps_tau:
            self._status = SolverStatus.Solved
        else:
            self._status = SolverStatus.IterationMaximum

        # Print termination reason.
        print(f'Terminated: {self.status.message}', flush=True)
