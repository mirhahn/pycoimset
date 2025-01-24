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
Naive quadratic penalty method for constrained optimization.
'''

from dataclasses import dataclass
from enum import Enum
import logging
import math
import signal
import time
from types import NotImplementedType
from typing import Callable, Generic, Optional, Self, TypeVar, assert_never

import numpy
from numpy.typing import ArrayLike, NDArray

from ...helpers.functionals import transform
from ...logging import TabularLogger
from ...step import SteepestDescentStepFinder
from ...typing import (
    Constraint,
    ErrorNorm,
    Functional,
    JSONSerializable,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)
from ...typing.functional import Operator
from ...typing.space import SignedMeasure
from ...util.signals import InterruptionFlag, interruptible_method
from ..unconstrained.eval import eval_instat, eval_rho
from ..unconstrained.solver import SolverParameters
from .eval import (
    eval_infeas,
    eval_pen_func,
    eval_pen_grad,
    make_component_eval,
    update_pen_grad,
)


__all__ = ['PenaltySolver']


Spc = TypeVar('Spc', bound=SimilaritySpace)


# Logger for debugging.
logger = logging.getLogger(__name__)


# Find timing function
if hasattr(time, 'process_time'):
    get_time = time.process_time
elif hasattr(time, 'perf_counter'):
    get_time = time.perf_counter
else:
    get_time = time.monotonic


# Formatting for NumPy arrays.
def ndarray_debug_format(a: NDArray):
    return numpy.array2string(
        a,
        precision=3,
        suppress_small=False,
        formatter={
            'float_kind': lambda x: numpy.format_float_scientific(
                x, precision=3
            )
        }
    )


class PenaltySolver(Generic[Spc]):
    '''
    Constrained optimization solver using a naive penalty method.

    Parameters
    ----------
    obj : Functional[Spc]
        Objective functional.
    *con : Constraint[Spc]
        Constraints.
    x0 : SimilarityClass[Spc], optional
        Initial solution. Defaults to empty class.
    mu : float, optional
        Initial penalty parameter. Defaults to `0.0`.
    err_wgt : array-like, optional
        Error weights for evaluation. Defaults to equal weight for all
        functionals.
    grad_err_wgt : array-like, optional
        Gradient error weights for evaluation. Defaults to `err_wgt`.
    step : UnconstrainedStepFinder[Spc], optional
        Step finder to be used. Defaults to a steepest descent step
        finder.
    param : PenaltySolver.Parameters
        Algorithmic parameters.
    **kwargs
        Additional algorithmic parameters.
    '''

    @dataclass
    class Parameters(SolverParameters, JSONSerializable):
        '''
        Algorithmic parameters for the penalty solver.

        Attributes
        ----------
        feas_tol : float
            Feasibility tolerance.

        relaxed_feas_tol : float
            Relaxed feasibility tolerance. Used for early penalty parameter
            increase.
        '''
        feas_tol: float = 1e-3
        relaxed_abstol: float | None = None
        margin_infeas: float = 0.1
        penalty_limit: float = 100.0

    class Status(Enum):
        _ignore_ = ['Message']

        Running = -1
        Solved = 0
        Infeasible = 1
        IterationMaximum = 2
        SmallStep = 3
        UserInterruption = 4

        Message: dict[Self, str]

        @property
        def is_running(self) -> bool:
            '''Solver is still continuing iteration.'''
            return self.value < 0

        @property
        def is_solved(self) -> bool:
            '''Solver has found a solution.'''
            return self.value == 0

        @property
        def is_error(self) -> bool:
            '''Solver has detected an error.'''
            return self.value > 0

        @property
        def message(self) -> str:
            '''Human-readable message.'''
            return type(self).Message.get(self, "(no message defined)")

    Status.Message = {
        Status.Running: "still running",
        Status.Solved: "solution found",
        Status.Infeasible: "local infeasibility detected",
        Status.IterationMaximum: "iteration maximum exceeded",
        Status.SmallStep: "step too small",
        Status.UserInterruption: "interrupted by user",
    }

    @dataclass
    class Stats:
        n_iter: int = 0
        n_reject: int = 0

    space: Spc
    f: list[Callable[[SimilarityClass[Spc], float], tuple[float, float]]]
    f_grad: list[
        Callable[
            [SimilarityClass[Spc], float],
            tuple[SignedMeasure[Spc], float]
        ]
    ]
    x: SimilarityClass[Spc]
    mu: float
    tr_rad: float

    _p: Parameters
    _step: UnconstrainedStepFinder[Spc]
    _norm: ErrorNorm
    _errwgt: NDArray
    _graderrwgt: NDArray

    status: Status
    stats: Stats
    logger: TabularLogger
    callback: Optional[Callable[[Self], None]]

    def __init__(self,
                 obj: Functional[Spc], *cons: Constraint[Spc],
                 x0: Optional[SimilarityClass[Spc]] = None,
                 mu: Optional[float] = None,
                 err_wgt: Optional[ArrayLike] = None,
                 grad_err_wgt: Optional[ArrayLike] = None,
                 step: Optional[UnconstrainedStepFinder[Spc]] = None,
                 param: Optional[Parameters] = None,
                 callback: Optional[Callable[[Self], None]] = None):
        # Set up initial solution object.
        def constr_func(con: Constraint[Spc]) -> Functional[Spc]:
            if (op := con.op) is Operator.EQUAL_TO:
                raise TypeError('Equality constraints are not supported')
            elif op is Operator.LESS_THAN:
                return transform(con.func, shift=-con.shift)
            elif op is Operator.GREATER_THAN:
                return transform(con.func, shift=con.shift, scale=-1)
            else:
                assert_never(op)
        self.f, self.f_grad, self._norm = make_component_eval(
            obj, *(constr_func(con) for con in cons), cache_size=2
        )

        # Set up remaining state
        self.space = obj.input_space
        self.x = x0 if x0 is not None else obj.input_space.empty_class
        self.mu = mu if mu is not None else 1.0
        self._p = param if param is not None else type(self).Parameters()
        self._step = (
            step if step is not None
            else SteepestDescentStepFinder[Spc]()
        )
        self._errwgt = numpy.broadcast_to(
            err_wgt
            if err_wgt is not None
            else 1.0,
            len(self.f)
        )
        self._graderrwgt = numpy.broadcast_to(
            grad_err_wgt
            if grad_err_wgt is not None
            else self._errwgt,
            len(self.f)
        )

        self.tr_rad = (
            tr_rad
            if (tr_rad := self._p.tr_radius) is not None
            else obj.input_space.measure
        )
        self.status = type(self).Status.Running
        self.stats = type(self).Stats()
        self.callback = callback

        # Set up logger.
        self.logger = TabularLogger(
            ['time', 'iter', 'objval', 'instat', 'infeas', 'penalty', 'step',
             'rejected'],
            format={
                'time': '8.2f',
                'iter': '4d',
                'objval': '13.6e',
                'instat': '13.6e',
                'infeas': '13.6e',
                'penalty': '13.6e',
                'step': '13.6e',
                'rejected': '4d'
            },
            width={
                'time': 8,
                'iter': 4,
                'objval': 13,
                'instat': 13,
                'infeas': 13,
                'penalty': 13,
                'step': 13,
                'rejected': 4
            },
            flush=True
        )

    def _callback(self) -> float:
        '''
        Run callback and return runtime.
        '''
        if (cb := self.callback) is not None:
            start_time = get_time()
            cb(self)
            return get_time() - start_time
        return 0.0

    @interruptible_method(signals=signal.SIGINT | signal.SIGTERM)
    def solve(self, int_flg: InterruptionFlag) -> None:
        '''Run main loop until termination.'''
        # Set up evaluators.
        fix_pen_val = eval_pen_func(self.f,
                                    val_wgt := self._errwgt,
                                    err_bnd=math.inf,
                                    err_decay=0.9)
        fix_pen_grad = eval_pen_grad(self.f,
                                     self.f_grad,
                                     self._norm,
                                     self._graderrwgt,
                                     grad_err_bnd=math.inf,
                                     grad_err_decay=0.9)
        get_instat = eval_instat(self.space, cache_size=1)
        pen_val = None
        pen_grad = None

        # Record start time.
        cb_time = 0.0
        start_time = get_time()

        # Evaluate initial infeasibility.
        nu, _ = eval_infeas(
            self.f[:-1],
            x := self.x,
            (p := self._p).margin_infeas,
            p.feas_tol,
            val_wgt[:-1],
            err_bnd=math.inf,
            err_decay=0.9
        )

        # Evaluate gradient and update penalty
        grad, _, mu, pen_grad = update_pen_grad(
            fix_pen_grad,
            self._norm,
            x,
            p.feas_tol,
            p.abstol,
            (eps_rel
             if (eps_rel := p.relaxed_abstol) is not None
             else 10 * p.abstol),
            p.margin_instat,
            p.margin_step,
            p.margin_proj_desc,
            (step := self._step).quality,
            nu,
            self.mu,
            p.penalty_limit,
            tr_rad := min(self.tr_rad, self.space.measure),
            pen_grad=pen_grad,
            instat_evaluator=get_instat,
            err_bnd=math.inf,
            err_decay=0.9
        )
        self.mu = mu
        set_neg, tau = get_instat(grad)
        pen_val = fix_pen_val(mu)

        # Output initial iterate.
        self.logger.push_line(
            time=get_time() - start_time - cb_time,
            iter=self.stats.n_iter,
            objval=self.f[-1](x, math.inf)[0],
            instat=tau,
            infeas=nu,
            penalty=mu,
        )
        self.status = type(self).Status.Running

        # Invoke callback if necessary.
        cb_time += self._callback()

        start_iter = self.stats.n_iter
        old_reject = self.stats.n_reject
        while int_flg.deferred_signal is None and (
            p.max_iter is None or self.stats.n_iter - start_iter < p.max_iter
        ) and get_instat(grad)[1] > p.abstol:
            # Get next step
            step.gradient = grad
            step.radius = tr_rad
            step.tolerance = (
                p.margin_step * step.quality
                * (tr_rad / max(tr_rad, set_neg.measure)) * tau
            )
            d, _ = step.get_step()

            # Calculate step quality
            xs = x ^ d
            if isinstance(xs, NotImplementedType):
                raise NotImplementedError(
                    'Symmetric difference not implemented'
                )
            (*_, f2, _, rho), err_rho = eval_rho(
                pen_val,
                grad,
                p.thres_accept,
                p.thres_reject,
                x,
                d,
                xs,
                err_bnd=math.inf,
                err_decay=0.9
            )

            # Decide whether to accept the step
            if (step_accept := rho - err_rho >= p.thres_accept):
                # Adopt new solution
                self.x = x = xs

                # Update stats
                self.stats.n_iter += 1

                # Update trust region radius
                if rho >= p.thres_tr_expand:
                    self.tr_rad = tr_rad = min(self.space.measure, 2 * tr_rad)

                # Evaluate new infeasibility.
                nu, _ = eval_infeas(
                    self.f[:-1],
                    x,
                    p.margin_infeas,
                    p.feas_tol,
                    val_wgt[:-1],
                    err_bnd=math.inf,
                    err_decay=0.9
                )
            else:
                self.tr_rad = tr_rad = tr_rad / 2
                self.stats.n_reject += 1

            # Update gradient
            grad, _, new_mu, pen_grad = update_pen_grad(
                fix_pen_grad,
                self._norm,
                x,
                p.feas_tol,
                p.abstol,
                (eps_rel
                 if (eps_rel := p.relaxed_abstol) is not None
                 else 10 * p.abstol),
                p.margin_instat,
                p.margin_step,
                p.margin_proj_desc,
                step.quality,
                nu,
                self.mu,
                p.penalty_limit,
                tr_rad,
                pen_grad=pen_grad,
                instat_evaluator=get_instat,
                err_bnd=math.inf,
                err_decay=0.9
            )
            set_neg, tau = get_instat(grad)
            if new_mu != mu:
                self.mu = mu = new_mu
                pen_val = fix_pen_val(mu)

            # Output log line and reset iteration rejection counter
            if step_accept:
                self.logger.push_line(
                    time=get_time() - start_time - cb_time,
                    iter=self.stats.n_iter,
                    objval=f2,
                    instat=tau,
                    infeas=nu,
                    penalty=mu,
                    step=d.measure,
                    rejected=self.stats.n_reject - old_reject
                )
                old_reject = self.stats.n_reject

                # Invoke callback if necessary.
                cb_time += self._callback()

        # Set status.
        if int_flg.deferred_signal is not None:
            self.status = type(self).Status.UserInterruption
        elif mu > p.penalty_limit:
            self.status = type(self).Status.Infeasible
        elif tau <= p.abstol:
            self.status = type(self).Status.Solved
        else:
            self.status = type(self).Status.IterationMaximum

        # Print termination reason.
        print(f'Terminated: {self.status.message}')
