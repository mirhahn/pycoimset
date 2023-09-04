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
Naive penalty method for constrained optimization.
'''

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
import logging
import math
import time
from typing import Any, Callable, Generic, Optional, Self, TypeVar, assert_never, cast

import numpy
from numpy import dtype, float_
from numpy.typing import ArrayLike, NDArray

from .unconstrained import SolverParameters
from ..logging import TabularLogger
from ..helpers import transform
from ..step import SteepestDescentStepFinder
from ..typing import (
    Constraint,
    ErrorNorm,
    Functional,
    JSONSerializable,
    Operator,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)
from ..util import depends_on, notify_property_update, tracks_dependencies


__all__ = ['PenaltySolver']


Spc = TypeVar('Spc', bound=SimilaritySpace)


# Logger for debugging.
logger = logging.getLogger(__name__)


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


@dataclass
class PenaltyFunctionals(Generic[Spc]):
    '''
    Collection of set functionals used by the penalty solver.
    '''
    obj: Functional[Spc]
    con_ub: Sequence[Functional[Spc]]
    con_eq: Sequence[Functional[Spc]]
    space: Spc = field(init=False)
    grad_tol_type: ErrorNorm = field(init=False)

    def __len__(self) -> int:
        '''Number of functionals.'''
        return 1 + len(self.con_ub) + len(self.con_eq)

    def __iter__(self) -> Generator[Functional[Spc], None, None]:
        '''Iterate over all functionals in canonical order.'''
        yield self.obj
        for f in self.con_ub:
            yield f
        for f in self.con_eq:
            yield f

    def __post_init__(self):
        self.space = self.obj.input_space
        self.grad_tol_type = self.obj.grad_tol_type
        for f in self:
            if f.input_space is not self.space:
                raise ValueError('input spaces disagree')
            if f.grad_tol_type is not self.grad_tol_type:
                raise ValueError('gradient error control types disagree')

    def errtol(self, tol: float, mu: float, *,
               con_max: Optional[ArrayLike] = None,
               wgt: ArrayLike = 1.0) -> numpy.ndarray:
        '''
        Distribute error tolerance over component functionals.

        Parameters
        ----------
        tol : float
            Total error tolerance.
        mu : float
            Penalty parameter.
        con_max: array-like, optional
            Maximum error tolerance for individual constraints. This is
            used to justify constraint violation assessments.
        wgt : array-like, optional
            Component error weights. Defaults to neutral weight.

        Returns
        -------
        numpy.ndarray
            Array of component error tolerances.

        Remarks
        -------
        Error weights may be ignored if normalization is considered
        numerically inaccurate.
        '''
        # Short-circuit for infinite tolerance.
        if not numpy.isfinite(tol):
            return numpy.full(len(self), numpy.inf)
        
        # Calculate modified weights.
        wgt_arr: numpy.ndarray[Any, dtype[float_]] = numpy.copy(numpy.broadcast_to(
            numpy.asarray(wgt, dtype=float), len(self)
        ))
        wgt_arr[1:] *= mu
        wgt_arr = abs(wgt_arr)

        if numpy.isclose((sum := wgt_arr.sum()), 0.0):
            wgt_arr = numpy.empty(len(self), dtype=float)
            wgt_arr[0] = 1.0
            wgt_arr[1:] = abs(mu)
            sum = wgt_arr.sum()

        etol = tol * (wgt_arr / sum)
        if con_max is not None:
            etol[1:] = numpy.fmin(
                etol[1:], numpy.broadcast_to(con_max, len(etol) - 1)
            )
            etol[0] = tol - numpy.sum(etol[1:])
        return etol


    @classmethod
    def from_problem(cls, obj: Functional[Spc], *con: Constraint[Spc]
                     ) -> Self:
        '''
        Generate functional collection from objective and constraint
        descriptions.
        '''
        con_ub = []
        con_eq = []

        for c in con:
            match c.op:
                case Operator.LESS_THAN:
                    con_ub.append(transform(c.func, shift=-c.shift))
                case Operator.GREATER_THAN:
                    con_ub.append(transform(c.func, scale=-1.0, shift=-c.shift))
                case Operator.EQUAL_TO:
                    con_eq.append(transform(c.func, shift=-c.shift))
                case _ as unreachable:
                    assert_never(unreachable)

        return cls(obj=obj, con_ub=con_ub, con_eq=con_eq)


@tracks_dependencies
class PenaltySolution(Generic[Spc]):
    '''
    Representation of a penalty solution.

    A solution consists of an argument alongside multiple evaluation
    results.

    Attributes
    ----------
    space
    func
    x
    mu
    val_wgt
    grad_wgt
    val_tol
    grad_tol
    con_tol
    val_err_tuple
    grad_err_tuple
    obj_val
    obj_grad
    pen_val
    pen_grad
    full_step
    tau
    '''
    _F: PenaltyFunctionals
    _x: Optional[SimilarityClass[Spc]]
    _mu: float
    _vt: float
    _gt: float
    _ct: NDArray[float_]
    _ve: NDArray[float_]
    _ge: NDArray[float_]

    def __init__(self, func: PenaltyFunctionals,
                 mu: Optional[float] = None,
                 x: Optional[SimilarityClass[Spc]] = None,
                 val_tol: Optional[float] = None,
                 grad_tol: Optional[float] = None,
                 con_tol: Optional[ArrayLike] = None,
                 val_wgt: Optional[ArrayLike] = None,
                 grad_wgt: Optional[ArrayLike] = None):
        self._F = func
        self._x = x
        self._mu = 0.0 if mu is None else mu
        self._vt = math.inf if val_tol is None else val_tol
        self._gt = math.inf if grad_tol is None else grad_tol
        self._ct = numpy.broadcast_to(
            math.inf if con_tol is None else con_tol,
            len(self._F) - 1
        )
        if val_wgt is None:
            val_wgt = 1.0 if grad_wgt is None else grad_wgt
        if grad_wgt is None:
            grad_wgt = val_wgt
        self._ve = numpy.broadcast_to(
            abs(numpy.asarray(val_wgt, dtype=float)), len(self._F)
        )
        self._ge = numpy.broadcast_to(
            abs(numpy.asarray(grad_wgt, dtype=float)), len(self._F)
        )

    def is_cache_invalid(self, prop: cached_property) -> bool:
        '''
        Check whether a given cached property is invalid.
        
        This is used in cache invalidation and is specific to the
        property in question. Currently, only `val_err_tuple` and
        `grad_err_tuple` are allowed.
        '''
        if prop is type(self).val_err_tuple:
            # Check whether there is a cached version.
            if not type(self).val_err_tuple.attrname in self.__dict__:
                return False
            
            # Get cached error values and derive error bound 
            val, val_err = self.val_err_tuple
            pen_err = val_err[0] + (self._mu / 2) * numpy.sum(
                (abs(val[1:]) + val_err[1:]) * val_err[1:]
            ).item()

            # Check whether error bound is violated.
            return (
                pen_err > self._vt
                or numpy.any(val_err[1:] <= self._ct).item()
            )
        elif prop is type(self).grad_err_tuple:
            # Check whether cache is available.
            if not type(self).grad_err_tuple.attrname in self.__dict__:
                return False
            
            # Invalidate cache if value-error tuple is no longer available.
            if not type(self).val_err_tuple.attrname in self.__dict__:
                return True
            
            # Get cached quantities.
            val, val_err = self.val_err_tuple
            _, grad_err = self.grad_err_tuple

            # Derive penalty gradient error bound.
            pen_grad_err = grad_err[0] + self._mu * (
                abs(val[1:]) + val_err[1:]
                ) * grad_err[1:]
            
            # Check error bound.
            return pen_grad_err > self._gt
        else:
            # Raise error if unknown property is passed.
            raise NotImplementedError()

    @property
    def space(self) -> Spc:
        '''Underlying similarity space.'''
        return self._F.obj.input_space
    
    @property
    def func(self) -> PenaltyFunctionals:
        '''Functional collection.'''
        return self._F

    @property
    def x(self) -> SimilarityClass[Spc]:
        '''Current argument.'''
        if self._x is None:
            self._x = self.space.empty_class
        return self._x

    @x.setter
    def x(self, x: SimilarityClass[Spc]) -> None:
        assert x.space is self.space
        self._x = x
        notify_property_update(self, 'x')

    @property
    def mu(self) -> float:
        '''Current penalty parameter.'''
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._mu = mu
        notify_property_update(self, 'mu')

    @property
    def val_wgt(self) -> NDArray[float_]:
        '''Value error weights.'''
        return self._ve

    @property
    def grad_wgt(self) -> NDArray[float_]:
        '''Gradient error weights.'''
        return self._ge

    @property
    def val_tol(self) -> float:
        '''Value tolerance.'''
        return self._vt

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        self._vt = tol
        notify_property_update(self, 'val_tol')

    @property
    def grad_tol(self) -> float:
        '''Gradient tolerance.'''
        return self._gt

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        self._gt = tol
        notify_property_update(self, 'grad_tol')

    @property
    def con_tol(self) -> NDArray[float_]:
        '''Constraint value tolerance.'''
        return self._ct

    @con_tol.setter
    def con_tol(self, tol: ArrayLike) -> None:
        tol = numpy.asarray(tol, dtype=float_)
        self._ct = cast(
            NDArray[float_], numpy.broadcast_to(tol, len(self._F) - 1)
        )
        notify_property_update(self, 'con_tol')

    @depends_on(
        x, mu,
        (val_tol, lambda s: s.is_cache_invalid(type(s).val_err_tuple)),
        (con_tol, lambda s: s.is_cache_invalid(type(s).val_err_tuple))
    )
    @cached_property
    def val_err_tuple(self) -> tuple[NDArray[float_], NDArray[float_]]:
        '''Value-error tuple.'''
        val_tol = self._F.errtol(self.val_tol, self.mu / 2, con_max=self._ct,
                                 wgt=self._ve)
        val = numpy.empty(len(self._F), dtype=float)
        err = numpy.empty(len(self._F), dtype=float)

        # Evaluate objective.
        if self._F.obj.arg is not self.x:
            self._F.obj.arg = self.x
        self._F.obj.val_tol = val_tol[0]
        val[0], err[0] = self._F.obj.get_value()

        # Clamping function for lower bound constraints.
        def val_ub_clamp(v: float, e: float) -> tuple[float, float]:
            if v < 0.0:
                e = max(0.0, v + e)
                v = 0.0
            return v, e

        # Evaluate upper bound constraints.
        idx_start = 0
        for con_seq, clamp_func in ((self._F.con_ub, val_ub_clamp),
                                    (self._F.con_eq, None)):
            for idx, (f, tol) in enumerate(
                zip(con_seq, val_tol[idx_start+1:]), start=idx_start + 1
            ):
                # Set argument.
                if f.arg is not self.x:
                    f.arg = self.x

                # Negotiate appropriate 'actual tolerance'.
                if self.mu == 0.0 or not numpy.isfinite(tol):
                    etol = numpy.inf
                else:
                    etol = numpy.sqrt(2 * tol / self.mu)
                while True:
                    # Perform evaluation.
                    f.val_tol = etol
                    v, e = f.get_value()

                    # Clamp value if necessary
                    if clamp_func is not None:
                        v, e = clamp_func(v, e)

                    # Terminate if tolerance is satisfied.
                    if (self.mu / 2) * (abs(v) + e) * e <= tol:
                        break

                    # Adjust actual tolerance.
                    etol = numpy.sqrt((abs(v) / 2)**2 + 2 * tol / self.mu) - abs(v) / 2
                    while (self.mu / 2) * (abs(v) + etol) * etol > tol:
                        etol *= 0.9

                # Store results.
                val[idx], err[idx] = v, e
            
            # Update start index.
            idx_start += len(con_seq)

        # Evaluate equality constraints.
        return val, err

    @depends_on(
        x, mu,
        (grad_tol, lambda s: s.is_cache_invalid(type(s).grad_err_tuple)),
        val_err_tuple
    )
    @cached_property
    def grad_err_tuple(self) -> tuple[Sequence[SignedMeasure[Spc]],
                                      NDArray[float_]]:
        '''Gradient-error tuple.'''
        # Get value-error tuple and calculate upper absolute value bound.
        val, verr = self.val_err_tuple
        vub = numpy.abs(val) + verr

        # Calculate component error tolerances.
        wgt = numpy.copy(self._ge)
        wgt[1:] *= vub[1:]
        grad_tol = self._F.errtol(self.grad_tol, self.mu, con_max=self._ct, wgt=wgt)

        # Set up output storage.
        grad = []
        err = numpy.empty(len(self._F), dtype=float)

        # Evaluate objective gradient.
        if self._F.obj.arg is not self.x:
            self._F.obj.arg = self.x
        self._F.obj.grad_tol = grad_tol[0]
        g, e = self._F.obj.get_gradient()
        grad.append(g)
        err[0] = e

        # Evaluate constraint gradients.
        idx_base = 0
        for con_seq in (self._F.con_ub, self._F.con_eq):
            for idx, (f, tol, ub) in enumerate(zip(
                con_seq, grad_tol[idx_base+1:], vub
            )):
                if f.arg is not self.x:
                    f.arg = self.x
                f.grad_tol = (math.inf if self.mu * ub == 0.0
                              else tol / (self.mu * ub))
                g, err[idx] = f.get_gradient()
                grad.append(g)
        return grad, err

    @property
    def obj_val(self) -> tuple[float, float]:
        '''Objective function value.'''
        val, err = self.val_err_tuple
        return val[0], err[0]

    @property
    def obj_grad(self) -> tuple[SignedMeasure[Spc], float]:
        '''Objective function gradient.'''
        grad, err = self.grad_err_tuple
        return grad[0], err[0]
    
    @depends_on(val_err_tuple, mu)
    @cached_property
    def pen_val(self) -> tuple[float, float]:
        '''Penalty function value.'''
        val, err = self.val_err_tuple
        return (val[0] + (self.mu / 2) * numpy.sum(val[1:]**2),
                err[0] + (self.mu / 2) * numpy.sum(
                    (2 * abs(val[1:]) + err[1:]) * err[1:]
                ))
    
    @depends_on(val_err_tuple, grad_err_tuple, mu)
    @cached_property
    def pen_grad(self) -> tuple[SignedMeasure[Spc], float]:
        '''Penalty function gradient.'''
        val, verr = self.val_err_tuple
        grad, gerr = self.grad_err_tuple
        grad_acc = grad[0]
        for v, g in zip(val[1:], grad[1:]):
            grad_acc = grad_acc + self.mu * v * g
        return grad_acc, gerr[0] + (self.mu * (abs(val[1:]) + verr[1:])
                                    * gerr[1:])
    
    @depends_on(pen_grad)
    @cached_property
    def full_step(self) -> SimilarityClass[Spc]:
        '''Full descent step for current penalty function.'''
        return self.pen_grad[0] < 0.0
    
    @depends_on(pen_grad, full_step)
    @cached_property
    def tau(self) -> tuple[float, float]:
        '''Instationarity of penalty function and error estimate.'''
        step = self.full_step
        grad, err = self.pen_grad
        return -grad(step), self._F.grad_tol_type.estimated_error(
            step.measure, err
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
            Feasibility tolerance per constraint.
        '''
        feas_tol: float = 1e-3

    class Status(Enum):
        _ignore_ = ['Message']

        Running          = -1
        Solved           = 0
        Infeasible       = 1
        IterationMaximum = 2
        SmallStep        = 3
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
        last_step: float = 0.0

    _sol: PenaltySolution[Spc]
    _par: Parameters
    _step: UnconstrainedStepFinder[Spc]
    _r: float
    _c: float

    status: Status
    stats: Stats
    logger: TabularLogger
    callback: Optional[Callable[[Self], None]]

    def __init__(self, obj: Functional[Spc], *con: Constraint[Spc],
                 x0: Optional[SimilarityClass[Spc]] = None,
                 mu: Optional[float] = None,
                 err_wgt: Optional[ArrayLike] = None,
                 grad_err_wgt: Optional[ArrayLike] = None,
                 step: Optional[UnconstrainedStepFinder[Spc]] = None,
                 param: Optional[Parameters] = None,
                 callback: Optional[Callable[[Self], None]] = None,
                 **kwargs):
        # Set up initial solution object.
        func = PenaltyFunctionals.from_problem(obj, *con)
        if x0 is None:
            x0 = func.space.empty_class
        if mu is None:
            mu = 0.0
        if err_wgt is None:
            err_wgt = 1.0
        if grad_err_wgt is None:
            grad_err_wgt = err_wgt
        self._sol = PenaltySolution(func, mu, x0, val_wgt=err_wgt,
                                    grad_wgt=grad_err_wgt)
        
        # Set up parameter object.
        if param is None:
            param = type(self).Parameters(**kwargs)
        else:
            param = replace(param, **kwargs)
        self._par = param

        # Set up step finder.
        if step is None:
            step = SteepestDescentStepFinder[Spc]()
        self._step = step

        # Set up additional variables.
        if self._par.tr_radius is None:
            self._r = self.space.measure
        else:
            self._r = min(self._par.tr_radius, self.space.measure)
        self._c = 0.0
        self.status = type(self).Status.Running
        self.stats = type(self).Stats()
        self.callback = callback

        # Set up logger.
        self.logger = TabularLogger(
            ['time', 'iter', 'objval', 'instat', 'infeas', 'penalty', 'step', 'rejected'],
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

    @property
    def space(self) -> Spc:
        '''Search space.'''
        return self._sol.space
    
    @property
    def solution(self) -> SimilarityClass[Spc]:
        '''Current solution.'''
        return self._sol.x
    
    @solution.setter
    def solution(self, x: SimilarityClass[Spc]) -> None:
        assert x.space is self._sol.space
        self._sol.x = x

    @property
    def penalty_parameter(self) -> float:
        '''Current penalty parameter.'''
        return self._sol.mu
    
    @penalty_parameter.setter
    def penalty_parameter(self, mu: float) -> None:
        self._sol.mu = mu

    @property
    def objective_functional(self) -> Functional[Spc]:
        '''Objective functional.'''
        return self._sol.func.obj
    
    @property
    def objective_value(self) -> float:
        '''Current objective function value.'''
        return self._sol.obj_val[0]
    
    @property
    def constraint_violation(self) -> numpy.ndarray:
        '''
        Current constraint violation.

        Remarks
        -------
        This is arranged in canonical order with inequality constraints
        preceding equality constraints. Otherwise, the order of the
        constraints is preserved.
        '''
        return self._sol.val_err_tuple[0][1:]
    
    @property
    def instationarity(self) -> float:
        '''Current instationarity of the penalty function.'''
        return self._sol.tau[0]
    
    def _tolerances(self,
                    tau: Optional[float] = None,
                    con: Optional[ArrayLike] = None,
                    rho: Optional[float] = None
                    ) -> tuple[float, float, float, numpy.ndarray]:
        '''
        Choose appropriate error tolerances.

        This method is used by the optimization loop to determine
        evaluation error margins for the objective functional. It uses
        prior guesses for instationarity and step quality.

        Parameters
        ----------
        tau : float, optional
            Anticipated unsigned instationarity.
        con : array-like, optional
            Anticipated constraint violations.
        rho : float, optional
            Anticipated step quality.

        Returns
        -------
        tuple[float, float, float, float]
            Penalty error tolerance, gradient error tolerance, step
            error tolerance, and constraint tolerances in that order.
        '''
        # Get parameters.
        eps = self._par.abstol
        eps_con = self._par.feas_tol
        xi_step = self._par.margin_step
        xi_prdesc = self._par.margin_proj_desc
        xi_instat = self._par.margin_instat
        step_quality = self._step.quality
        radius = self._r
        mu_full = self._sol.full_step.measure
        norm = self._sol.func.grad_tol_type

        # Guess instationarity if no guess is given.
        if tau is None: 
            tau = max(eps, self._sol.tau[0])
        else:
            tau = max(eps, tau)

        # Approximate upper bound to full step measure.
        mu_full_ub = self.space.measure
        if norm is ErrorNorm.Linfty:
            grad, grad_err = self._sol.pen_grad
            mu_full_ub = (grad < grad_err).measure

        # Calculate lower bound for projected descent.
        expected_step_ratio = step_quality * min(
            1.0, radius / mu_full
        )
        proj_desc_min = (1.0 - xi_step) * expected_step_ratio * tau

        # Guess step quality if no guess is given
        if rho is None:
            if numpy.isclose(proj_desc_min, 0.0): 
                rho = 1.0
            else:
                rho = 1.0 - (
                    self._c * min(radius, mu_full)**2 / (2 * proj_desc_min)
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
        sigma_0 = self._par.thres_accept
        sigma_1 = self._par.thres_reject
        rho_tol = max(rho - sigma_0, sigma_1 - rho, (sigma_1 - sigma_0) / 4)
        obj_tol = rho_tol * proj_desc_min / 2

        # Calculate step tolerance.
        step_tol = xi_step * expected_step_ratio * tau

        # Calculate constraint violation error tolerance.
        if con is not None:
            con = numpy.asarray(con, dtype=float)
            con_tol = numpy.fmax(eps_con, numpy.abs(con - eps_con)) / 2
        else:
            con_tol = numpy.full(len(self._sol.func) - 1, numpy.inf)

        # Log tolerances (debugging).
        for name, val in (('obj_tol', obj_tol),
                          ('grad_tol', grad_tol),
                          ('step_tol', step_tol),
                          ('con_tol', con_tol)):
            if isinstance(val, float):
                logger.getChild('tolerances').debug(f'{name} = {val:.3e}')
            elif isinstance(val, numpy.ndarray):
                val = numpy.array2string(val, precision=3, floatmode='fixed')
                logger.getChild('tolerances').debug(f'{name} = {val}')
            else:
                assert_never(val)

        return obj_tol, grad_tol, step_tol, con_tol

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
            self._c = 0.9 * self._c + 0.1 * curvature
    
    def step(self) -> None:
        '''
        Perform a single step of the penalty main loop.

        A single step consists of the following parts:

        1. Check for epsilon stationarity and constraint satisfaction.
           Terminate if both conditions are met.
        2. Find a descent step with respect to the current penalty
           function.
        3. Accept the step if it is sufficiently accurate.
        4. Increase penalty parameter if any constraint violations were
           increased by the step.
        '''
        # Obtain instationarity and instationarity error as well as
        # constraint violations.
        while True:
            tau, tau_err = self._sol.tau
            val, val_err = self._sol.val_err_tuple

            obj_tol, grad_tol, step_tol, con_tol = self._tolerances(
                tau=tau,
                con=val[1:]
            )
            if (
                tau_err <= self._par.margin_instat * max(
                    tau - self._par.abstol, self._par.abstol
                )
                and numpy.all(val_err[1:] <= con_tol)
            ):
                break
        
            self._sol.val_tol = obj_tol
            self._sol.grad_tol = grad_tol
            self._sol.con_tol = con_tol

        # Check termination criterion.
        if (tau <= self._par.abstol
            and numpy.all(val[1:] <= self._par.feas_tol)):
            self.status = type(self).Status.Solved
            return
        
        accepted = False
        comp_proj_chg = None
        while not accepted:
            # Find step.
            grad, grad_err = self._sol.pen_grad
            self._step.gradient = grad
            self._step.radius = self._r
            self._step.tolerance = step_tol
            step, _ = self._step.get_step()

            # Calculate projected change on a per-functional basis.
            comp_val, _ = self._sol.val_err_tuple
            comp_grad, _ = self._sol.grad_err_tuple
            comp_proj_chg = numpy.array([g(step) for g in comp_grad], dtype=float_)
            logger.getChild('step').debug(f'mu = {self._sol.mu}')
            logger.getChild('step').debug(
                'comp_val = '
                f'{ndarray_debug_format(comp_val)}'
            )
            logger.getChild('step').debug(
                'comp_proj_chg = '
                f'{ndarray_debug_format(comp_proj_chg)}'
            )

            # Aggregate full projected change.
            proj_chg = grad(step)
            proj_chg_error = self._sol.func.grad_tol_type.estimated_error(
                step.measure, grad_err
            )
            if proj_chg_error > self._par.margin_proj_desc * abs(proj_chg):
                val, _ = self._sol.val_err_tuple
                obj_tol, grad_tol, step_tol, con_tol = self._tolerances(
                    tau=tau, con=val[1:], rho=proj_chg / step.measure
                )
                self._sol.val_tol = obj_tol
                self._sol.grad_tol = grad_tol
                self._sol.con_tol = con_tol
                continue

            # Calculate step quality.
            new_sol = PenaltySolution[Spc](
                self._sol.func,
                self._sol.mu,
                self._sol.x ^ step,
                val_tol=obj_tol,
                grad_tol=grad_tol,
                con_tol=con_tol,
                val_wgt=self._sol.val_wgt,
                grad_wgt=self._sol.grad_wgt
            )
            pen, pen_err = self._sol.pen_val
            new_pen, new_pen_err = new_sol.pen_val
            step_quality = (new_pen - pen) / proj_chg
            step_quality_err = (pen_err + new_pen_err) / abs(proj_chg)

            # Log actual component change.
            comp_chg = new_sol.val_err_tuple[0] - self._sol.val_err_tuple[0]
            logger.getChild('step').debug(f'comp_chg = {ndarray_debug_format(comp_chg)}')

            # Update curvature estimate.
            self._update_curvature(step.measure, proj_chg, new_pen - pen)

            # Update penalty parameter if necessary.
            viol_old = self._sol.val_err_tuple[0][1:]
            viol_chg = numpy.sum(new_sol.val_err_tuple[0][1:] - self._sol.val_err_tuple[0][1:])

            # Decide whether or not to accept.
            if step_quality - step_quality_err >= self._par.thres_accept:
                # Accept new solution.
                self._sol = new_sol

                # Increment iteration counter.
                self.stats.n_iter += 1
                self.stats.last_step = step.measure

                # Increase TR radius if necessary.
                if step_quality >= self._par.thres_tr_expand:
                    self._r = min(self.space.measure, 2 * self._r)

                # Update penalty parameter if necessary.
                if viol_chg > 0 and numpy.any(viol_old > self._par.feas_tol):
                    self._sol.mu = 2 * self._sol.mu
                    self._c = 0.0

                accepted = True
            elif step_quality + step_quality_err < self._par.thres_reject:
                # Log decision.
                logger.getChild('step').debug(
                    'rejected step with rho <= '
                    f'{step_quality + step_quality_err:.3f}'
                )

                # Halve radius.
                self._r = self._r / 2
                logger.getChild('step').info(f'radius = {self._r} (rho = {step_quality})')

                # Terminate if radius is too small.
                if self._r < 1000 * numpy.finfo(float).eps:
                    self.status = type(self).Status.SmallStep

                # Increase rejection counter.
                self.stats.n_reject += 1

            # Adjust error tolerances.
            obj_tol, grad_tol, step_tol, con_tol = self._tolerances(
                tau=tau,
                con=val[1:],
                rho=step_quality
            )
            self._sol.val_tol = obj_tol
            self._sol.grad_tol = grad_tol
            self._sol.con_tol = con_tol

    def solve(self) -> None:
        '''Run main loop until termination.'''
        # Record start time.
        start_time = time.perf_counter()

        # Output initial iterate.
        self.logger.push_line(
            time=time.perf_counter() - start_time,
            iter=self.stats.n_iter,
            objval=self._sol.obj_val[0],
            instat=self._sol.tau[0],
            infeas=numpy.sum(self._sol.val_err_tuple[0][1:]),
            penalty=self._sol.mu,
        )

        # Invoke callback if necessary.
        if self.callback is not None:
            self.callback(self)

        self.status = type(self).Status.Running
        start_iter = self.stats.n_iter
        while (self.status.is_running and (
            self._par.max_iter is None
            or self.stats.n_iter - start_iter < self._par.max_iter
        )):
            # Remember number of previously rejected steps.
            old_reject = self.stats.n_reject

            # Perform a step
            self.step()

            # Output log line.
            self.logger.push_line(
                time=time.perf_counter() - start_time,
                iter=self.stats.n_iter,
                objval=self._sol.obj_val[0],
                instat=self._sol.tau[0],
                infeas=numpy.sum(self._sol.val_err_tuple[0][1:]),
                penalty=self._sol.mu,
                step=self.stats.last_step,
                rejected=self.stats.n_reject - old_reject
            )

            # Invoke callback if necessary.
            if self.callback is not None:
                self.callback(self)
        
        # Set status if number of iterations was exceeded.
        if self.status is type(self).Status.Running:
            self.status = type(self).Status.IterationMaximum

        # Print termination reason.
        print(f'Terminated: {self.status.message}')