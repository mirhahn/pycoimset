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
Log-barrier method solver.
'''

from dataclasses import dataclass
import dataclasses
from enum import Enum
from functools import cached_property
import logging
import math
from typing import Generator, Generic, Optional, Sequence, TypeVar

import numpy
from numpy.typing import ArrayLike

from .unconstrained import SolverParameters as UnconstrainedParameters
from ..step import SteepestDescentStepFinder
from ..helpers import transform
from ..logging import TabularLogger
from ..typing import (
    Constraint,
    Functional,
    Operator,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)
from ..util import depends_on, notify_property_update, tracks_dependencies


__all__ = [
    'BarrierFunctionals',
    'BarrierSolution',
    'BarrierSolver',
]

# Module logger.
logger = logging.getLogger('pycoimset.solver.barrier')


# Type variable for similarity space type.
Spc = TypeVar('Spc', bound=SimilaritySpace)


@dataclass(frozen=True)
class BarrierFunctionals(Generic[Spc]):
    obj: Functional[Spc]
    con: list[Functional[Spc]]
    vwgt: Optional[numpy.ndarray] = None
    gwgt: Optional[numpy.ndarray] = None

    def __len__(self) -> int:
        '''Total number of functionals.'''
        return 1 + len(self.con)

    def comp_tol(self, tol: float, mu: float, wgt: Optional[ArrayLike] = None) -> numpy.ndarray:
        '''
        Distribute error tolerance among functionals.

        Parameters
        ----------
        tol : float
            Total tolerance to be distributed.
        mu : float
            Barrier parameter. Must be strictly positive.
        wgt : array-like, optional
            Vector of functional weights for error tolerance
            apportionment. Must be broadcastable to a 1-D array with
            one element per functional.

        Returns
        -------
        numpy.ndarray
            Array of length `len(self)` with error tolerances.

        Remarks
        -------
        The elements of the output array is arranged in the same way
        as is yielded by `self.functionals` with the objective
        functional corresponding to the first entry.

        For the constraint functionals, this yields the error
        apportioned to the logarithm of the functional value.
        '''
        if not numpy.isfinite(tol):
            return numpy.broadcast_to(tol, len(self))
        if wgt is None:
            wgt = 1.0
        wgt = numpy.abs(numpy.broadcast_to(wgt, len(self)))
        wgt[1:] *= mu
        wgt_sum = numpy.sum(wgt)
        return (wgt / wgt_sum) * tol 

    def functionals(self) -> Generator[Functional[Spc], None, None]:
        '''Iterates over all functionals.'''
        yield self.obj
        for c in self.con:
            yield c


@tracks_dependencies
class BarrierSolution(Generic[Spc]):
    _f: BarrierFunctionals[Spc]
    _x: Optional[SimilarityClass[Spc]]
    _mu: float
    _tolf: float
    _tolg: float
    _eps: float

    def __init__(self, func: BarrierFunctionals[Spc],
                 x: Optional[SimilarityClass[Spc]] = None,
                 mu: float = 1.0,
                 val_tol: float = math.inf,
                 grad_tol: float = math.inf,
                 eps: float = 1e-3):
        self._f = func
        self._x = x
        self._mu = mu
        self._tolf = val_tol
        self._tolg = grad_tol
        self._eps = eps

    def verify_value_errors(self, val: Optional[numpy.ndarray] = None,
                            err: Optional[numpy.ndarray] = None) -> bool:
        '''
        Verify whether a given evaluation result satisfies error
        bounds.

        Parameters
        ----------
        val : numpy.ndarray, optional
            Array of functional values. Taken from cache if either
            `val` or `err` are unspecified.
        err : numpy.ndarray, optional
            Array of functional errors. Taken from cache if either
            `val` or `err` are unspecified.

        Returns
        -------
        bool
            `True` if the evaluation result satisfies the error bounds.
            `False` if no data is given or the data does not satisfy
            the error bounds.

        Raises
        ------
        ValueError
            Data was provided, but has the wrong dimensions.

        Remarks
        -------
        The error bounds validated by this function are less strict
        than those enforced by the evaluation routine. They are
        considered satisfied if

            1. every constraint functional with `val - err <= 0`
               satisfies `val + err <= eps`;
            2. if `val - err > 0` for all constraint functionals, then
               each constraint functional satisfies `err <= val / 2`;
            3. if `val - err > 0` for all constraint functionals, then
               the aggregate error of the barrier function value is
               less than `val_tol`.

        Notably, it does not validate the component-wise error bounds
        enforced by the evaluation method.
        '''
        # Retrieve cached data if none is provided.
        if val is None or err is None:
            val, err = self.__dict__.get('f', (None, None))
        if val is None or err is None:
            return False

        # Check dimensions.
        if val.shape != (len(self._f),):
            raise ValueError('`val` has unsupported shape '
                             f'{val.shape} != {(len(self._f),)}')
        if err.shape != (len(self._f),):
            raise ValueError('`err` has unsupported shape '
                             f'{err.shape} != {(len(self._f),)}')

        # Detect apparent non-positive constraint values.
        conlb = val[1:] - err[1:]
        conub = val[1:] + err[1:]
        nonpos = conlb <= 0.0
        if numpy.any(nonpos):
            return bool(numpy.all(conub[nonpos] <= self._eps))

        # Ensure relative error of constraint functionals.
        if numpy.any(err[1:] > val[1:] / 2):
            return False

        # Calculate low and high errors on barrier function terms.
        err_low = numpy.array(err)
        err_low[1:] = numpy.log(val[1:] / conlb)
        err_high = numpy.array(err)
        err_high[1:] = numpy.log(conub / val[1:])

        # Calculate absolute error estimate for barrier function.
        err_bar = max(
            err_high[1:] + self._mu * err_low[1:].sum(),
            err_low[1:] + self._mu * err_high[1:].sum()
        )
        return err_bar <= self._tolf


    def verify_gradient_errors(self, fval: Optional[numpy.ndarray] = None,
                               ferr: Optional[numpy.ndarray] = None,
                               gerr: Optional[numpy.ndarray] = None) -> bool:
        '''
        Verify whether a given gradient evaluation result satisfies the
        error bound.

        Parameters
        ----------
        fval : numpy.ndarray, optional
            Functional values. Set to cached output if `fval`, `ferr`,
            or `gerr` are unspecified.
        ferr : numpy.ndarray, optional
            Functional value error estimates. Set to cached output if
            `fval`, `ferr`, or `gerr` are unspecified.
        gerr : numpy.ndarray, optional
            Functional gradient error estimates. Set to cached output
            if `fval`, `ferr`, or `gerr` are unspecified.

        Returns
        -------
        `True` if both functional values and gradients satisfy their
        respective error bounds. `False` if no data is available, if
        the data does not satisfy the error bound, or if there exists
        a constraint functional with a potentially non-positive output.

        Raises
        ------
        ValueError
            Indicates a disagreement in dimensions.

        Remarks
        -------
        Gradient error bounds are considered unsatisfied if the value
        error bounds are unsatisfied because this can lead to errors
        when calculating error bounds for the gradient of a logarithmic
        error term.
        '''
        # Retrieve data if not specified.
        if fval is None or ferr is None or gerr is None:
            fval, ferr = self.__dict__.get('f', (None, None))
            _, gerr = self.__dict__.get('g', (None, None))
        if fval is None or ferr is None or gerr is None:
            return False

        # Verify dimensions.
        nfunc = len(self._f)
        for arr in (fval, ferr, gerr):
            if arr.shape != (nfunc,):
                raise ValueError('shapes of evaluation results disagree with '
                                 f'expected shape {(nfunc,)}')

        # Ensure validity of functional values.
        if not self.verify_value_errors(fval, ferr):
            return False

        # Calculate error bound for barrier term.
        conlb = fval[1:] - ferr[1:]
        err = gerr[0] + self._mu * numpy.sum(gerr[1:] / conlb)

        return err <= self._tolg

    @property
    def x(self) -> SimilarityClass[Spc]:
        '''Argument.'''
        if self._x is None:
            self._x = self._f.obj.input_space.empty_class
        return self._x

    @x.setter
    def x(self, x: SimilarityClass[Spc]) -> None:
        if x is self._x:
            return
        self._x = x
        notify_property_update(self, 'x')

    @property
    def mu(self) -> float:
        '''Barrier parameter.'''
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._mu = mu
        notify_property_update(self, 'mu')

    @property
    def eps(self) -> float:
        '''Tolerance for detection of non-interior point.'''
        return self._eps

    @eps.setter
    def eps(self, eps: float) -> None:
        self._eps = eps
        notify_property_update(self, 'eps')

    @property
    def val_tol(self) -> float:
        '''Functional value error tolerance.'''
        return self._tolf

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        self._tolf = tol
        notify_property_update(self, 'val_tol')

    @property
    def grad_tol(self) -> float:
        '''Functional gradient error tolerance.'''
        return self._tolg

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        '''Functional gradient error tolerance.'''
        self._tolg = tol
        notify_property_update(self, 'grad_tol')

    @property
    def func(self) -> BarrierFunctionals[Spc]:
        '''Functionals.'''
        return self._f

    @depends_on(
        x,
        mu,
        (eps, lambda q: not q.verify_value_errors()),
        (val_tol, lambda q: not q.verify_value_errors())
    )
    @cached_property
    def f(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        Tuple of functional values and errors.

        Remarks
        -------
        Error control is performed with the expectation that only
        logarithms of the values will be used. Thus, error control
        will be performed such that either:

            1. the total error of the barrier function is less than the
               given value tolerances, or
            2. the constraint residual is ascertained to be less than a
               given `eps` (> 0) that is deemed indistinguishable from
               zero.

        Additionally, for functional values strictly greater than
        `eps`, it is ascertained that the error is less than half of
        the functional value. This ensures that the effect of the value
        error on the logarithm's derivative can halve the gradient
        tolerance at worst.
        '''
        tol = self._f.comp_tol(self._tolf, self._mu, self._f.vwgt)
        val = numpy.empty_like(tol)
        err = numpy.empty_like(tol)

        # Evaluate objective functional.
        if self._f.obj.arg is not self._x:
            self._f.obj.arg = self._x
        self._f.obj.val_tol = tol[0]
        val[0], err[0] = self._f.obj.get_value()

        # Evaluate constraint functionals.
        reltol = numpy.fmin(numpy.expm1(tol[1:]), -numpy.expm1(-tol[1:]))
        reltol = numpy.clip(reltol, None, 0.5)

        for idx, (rtol, f) in enumerate(zip(reltol, self._f.con), start=1):
            # Set argument and perform initial evaluation.
            if f.arg is not self._x:
                f.arg = self._x
            f.val_tol = math.inf
            v, e = f.get_value()

            while ((v - e <= 0.0 and v + e > self._eps)
                   or (v - e > 0.0 and e > v * rtol)):
                # Distinguish between separation case and refinement case.
                if v - e <= 0.0:
                    # Separation case: Separate `v` from either 0 or
                    # `eps`, whichever is further.
                    f.val_tol = max(self._eps - v, v) / 2
                else:
                    # Refinement case: `v` > 0, so we can apply rtol.
                    f.val_tol = v * rtol / 2
                v, e = f.get_value()

            # Store results.
            val[idx] = v
            err[idx] = e

        return val, err

    @depends_on(
        x, f, mu, (grad_tol, lambda q: not q.verify_gradient_errors())
    )
    @cached_property
    def g(self) -> tuple[Sequence[SignedMeasure[Spc]], numpy.ndarray]:
        '''Tuple of functional gradients and error estimates.'''
        # Retrieve values and error estimates. Calculate lower bound on
        # functional values for constraints.
        val, err = self.f
        conlb = val[1:] - err[1:]

        # Raise exception if solution appears non-interior.
        if numpy.any(conlb <= 0.0):
            raise ValueError('solution appears non-interior '
                             f'({numpy.sum(conlb <= 0.0)} '
                             'violations)')

        # Apportion tolerances.
        tol = self._f.comp_tol(self._tolg, self._mu, self._f.gwgt)

        # Set up output.
        grad = []
        err = numpy.empty(len(self._f))

        # Evaluate objective gradient.
        if self._f.obj.arg is not self._x:
            self._f.obj.arg = self._x
        self._f.obj.grad_tol = tol[0]
        m, err[0] = self._f.obj.get_gradient()
        grad.append(m)

        # Evaluate constraint functional gradients.
        for idx, (f, vl) in enumerate(zip(self._f.con, conlb), start=1):
            if f.arg is not self._x:
                f.arg = self._x
            f.grad_tol = tol[idx] * vl
            m, err[idx] = f.get_gradient()
            grad.append(m)

        return grad, err

    @depends_on(f, mu)
    @cached_property
    def dual_multipliers(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''Approximation of Lagrange multipliers.'''
        val, valerr = self.f
        if numpy.any(val - valerr <= 0):
            raise ValueError('possible non-interior solution detected')
        mult = -self.mu / val
        multerr = self.mu * valerr / (val * (val - valerr))
        return mult, multerr

    @depends_on(f, mu)
    @cached_property
    def F(self) -> tuple[float, float]:
        '''Barrier function value.'''
        # Get component values and check if any constraints are
        # non-interior.
        val, err = self.f
        if (nerr := numpy.sum(val[1:] - err[1:] <= 0.0)) > 0:
            raise ValueError('solution appears non-interior '
                             f'({nerr} violations)')

        # Calculate weighted sum.
        err = err[0] + self._mu * numpy.sum(numpy.log(numpy.fmax(
            val[1:] / (val[1:] - err[1:]),
            (val[1:] + err[1:]) / val[1:]
        )))
        val = val[0] - self._mu * numpy.sum(numpy.log(val[1:]))

        return val, err

    @depends_on(f, g, mu)
    @cached_property
    def G(self) -> tuple[SignedMeasure[Spc], float]:
        '''Barrier function value.'''
        # Get component values and check if any constraints are
        # non-interior.
        gval, gerr = self.g
        fval, ferr = self.f
        conlb = fval[1:] - ferr[1:]
        if (nerr := numpy.sum(conlb <= 0.0)) > 0:
            raise ValueError('solution appears non-interior '
                             f'({nerr} violations)')

        # Calculate weighted sum.
        err = gerr[0] + self._mu * numpy.sum(gerr[1:] / conlb)
        val = sum(
            ((-self._mu / v) * g for g, v in zip(gval[1:], fval[1:])),
            start=gval[0]
        )

        return val, err

    @depends_on(G)
    @cached_property
    def full_step(self) -> SimilarityClass[Spc]:
        '''Full descent step at current solution.'''
        grad, _ = self.G
        return grad < 0.0

    @depends_on(G, full_step)
    @cached_property
    def instationarity(self) -> tuple[float, float]:
        '''Instationarity and error at current solution.'''
        grad, err = self.G
        val = grad(self.full_step)
        valerr = self.func.obj.grad_tol_type.estimated_error(
            self.full_step.measure, err
        )
        return -val, valerr


class BarrierSolver(Generic[Spc]):
    '''
    Basic log-barrier solver.
    '''
    @dataclass
    class Parameters(UnconstrainedParameters):
        '''
        Parameters for log-barrier method.
        '''
        #: Interior margin for constraints.
        con_eps: float = 1e-6

        #: Complementarity tolerance.
        compltol: float = 1e-3

        #: Initial barrier parameter
        mu_init: float = 1.0

        #: Exponential of decay rate for the barrier parameter.
        mu_decay: float = 0.9

    @dataclass
    class Stats:
        '''
        Stats collected by the solver.
        '''
        #: Number of completed iterations.
        n_iter: int = 0

        #: Number of rejected steps.
        n_reject: int = 0

        #: Measure of last step.
        last_step: float = 0.0

    class Status(Enum):
        _ignore_ = ['Message']

        Running          = -1
        Solved           = 0
        Infeasible       = 1
        IterationMaximum = 2
        SmallStep        = 3
        UserInterruption = 4

        Message: 'dict[BarrierSolver.Status, str]'

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
        Status.Infeasible: "non-interior solution detected",
        Status.IterationMaximum: "iteration maximum exceeded",
        Status.SmallStep: "step too small",
        Status.UserInterruption: "interrupted by user",
    }

    _par: Parameters
    _status: Status
    _sol: BarrierSolution[Spc]
    _step: UnconstrainedStepFinder
    _stats: Stats
    _log: TabularLogger
    _r: float
    _c: float

    def __init__(self, obj: Functional[Spc],
                 con: Sequence[Constraint[Spc]],
                 err_wgt: Optional[ArrayLike] = None,
                 grad_err_wgt: Optional[ArrayLike] = None,
                 x0: Optional[SimilarityClass[Spc]] = None,
                 param: Optional[Parameters] = None,
                 *args, **kwargs):
        # Import parameters.
        if param is None:
            self._par = type(self).Parameters(*args, **kwargs)
        else:
            args = {
                f.name: v for v, f in zip(
                    args, dataclasses.fields(type(self).Parameters)
                )
            }
            self._par = dataclasses.replace(param, **args, **kwargs)

        # Bring constraints into suitable form.
        con_func = []
        for c in con:
            if c.func.grad_tol_type is not obj.grad_tol_type:
                raise ValueError('component functionals must have same type '
                                 'of gradient error control')
            if c.func.input_space is not obj.input_space:
                raise ValueError('component functionals must be defined over '
                                 'same input space')
            if c.op is Operator.EQUAL_TO:
                raise ValueError('barrier method cannot handle equality '
                                 'constraints')
            if c.op is Operator.GREATER_THAN:
                con_func.append(
                    transform(c.func, shift=-c.shift)
                )
            elif c.op is Operator.LESS_THAN:
                con_func.append(
                    transform(c.func, scale=-1, shift=c.shift)
                )
            else:
                raise ValueError(f'unknown constraint operator {c.op}')

        # Create default solution.
        if x0 is None:
            x0 = obj.input_space.empty_class
        if err_wgt is not None:
            err_wgt = numpy.broadcast_to(numpy.asarray(err_wgt, dtype=float),
                                         1 + len(con_func))
        if grad_err_wgt is not None:
            grad_err_wgt = numpy.broadcast_to(
                numpy.asarray(grad_err_wgt, dtype=float),
                1 + len(con_func)
            )
        elif err_wgt is not None:
            grad_err_wgt = err_wgt
        self._sol = BarrierSolution[Spc](
            BarrierFunctionals[Spc](
                obj,
                con_func,
                vwgt=err_wgt,
                gwgt=grad_err_wgt
            ),
            x0,
            mu=self._par.mu_init,
            eps=self._par.con_eps
        )

        # Set up logger.
        self._log = TabularLogger(
            ['iter', 'objval', 'instat', 'barrier', 'step', 'rejected'],
            format={
                'iter': '4d',
                'objval': '13.6e',
                'instat': '13.6e',
                'barrier': '13.6e',
                'step': '13.6e',
                'rejected': '4d'
            },
            width={
                'iter': 4,
                'objval': 13,
                'instat': 13,
                'barrier': 13,
                'step': 13,
                'rejected': 4
            },
            flush=True
        )

        # Set remaining variables.
        self._status = type(self).Status.Running
        self._stats = type(self).Stats()
        self._step = SteepestDescentStepFinder()
        self._r = min(self._par.tr_radius, x0.space.measure)
        self._c = 0.0

    def _errtol(self, tau: Optional[float] = None, rho: Optional[float] = None
                ) -> tuple[float, float, float]:
        # Get parameters.
        eps = self._par.abstol
        xi_step = self._par.margin_step
        xi_prdesc = self._par.margin_proj_desc
        xi_instat = self._par.margin_instat
        step_quality = self._step.quality
        radius = self._r
        mu_full = self._sol.full_step.measure
        norm = self._sol.func.obj.grad_tol_type

        # Guess instationarity if no guess is given.
        if tau is None:
            tau = self._sol.instationarity[0]
        tau = max(eps, tau)

        # Fall back to measure of universal set if full step appears
        # empty.
        if numpy.isclose(mu_full, 0.0):
            mu_full = self._sol.func.obj.input_space.measure

        # Calculate lower bound for projected descent.
        expected_step_ratio = step_quality * min(
            1.0, radius / mu_full
        )
        proj_desc_min = (1.0 - xi_step) * expected_step_ratio * tau

        # Guess step quality if no guess is given
        if rho is None:
            rho = 1.0 - (
                self._c * min(radius, mu_full)**2 / (2 * proj_desc_min)
            )

        # Calculate gradient error tolerance.
        grad_tol_step = norm.required_tolerance(
            radius, xi_prdesc * proj_desc_min
        )
        grad_tol_full = norm.required_tolerance(
            mu_full, xi_instat * max(tau - eps, eps)
        )
        grad_tol = min(grad_tol_step, grad_tol_full) / 2

        # Calculate objective error tolerance.
        sigma_0 = self._par.thres_accept
        sigma_1 = self._par.thres_reject
        rho_tol = max(rho - sigma_0, sigma_1 - rho, (sigma_1 - sigma_0) / 4)
        obj_tol = rho_tol * proj_desc_min / 2

        # Calculate step tolerance.
        step_tol = xi_step * expected_step_ratio * tau

        return obj_tol, grad_tol, step_tol

    def _updcurv(self, step_size: float, projected_descent: float,
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

    @property
    def x(self) -> SimilarityClass[Spc]:
        '''Current solution.'''
        return self._sol.x

    @x.setter
    def x(self, x: SimilarityClass[Spc]) -> None:
        if x is self._sol.x:
            return
        self._sol.x = x

    @property
    def z(self) -> numpy.ndarray:
        '''Current dual multiplier estimate.'''
        return self._sol.dual_multipliers[0]

    @property
    def mu(self) -> float:
        '''Barrier parameter.'''
        return self._sol.mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._sol.mu = mu

    def step(self) -> None:
        '''
        Perform single step.
        '''
        # Check if current solution is interior.
        val, err = self._sol.f
        if numpy.any(val[1:] - err[1:] <= 0.0):
            self._status = type(self).Status.Infeasible
            return

        # Update barrier parameter as needed.
        mu_pre = self._sol.mu
        while True:
            # Calculate instationarity.
            while True:
                tau, tau_err = self._sol.instationarity
                bar_tol, grad_tol, step_tol = self._errtol(tau)
                if tau_err <= self._par.margin_instat * max(
                    tau - self._par.abstol, self._par.abstol
                ):
                    break
                self._sol.val_tol = bar_tol
                self._sol.grad_tol = grad_tol

            # Perform barrier function instationarity test.
            if tau > self._par.abstol:
                break

            # Perform complementarity test.
            if self._sol.mu <= self._par.compltol:
                self._status = type(self).Status.Solved
                return

            # Decrease barrier parameter.
            self._sol.mu = self._sol.mu * self._par.mu_decay
        if self._sol.mu != mu_pre:
            logger.info(f'`mu` lowered to {self._sol.mu:0.3g}')
        del mu_pre

        # Perform next step.
        accepted = False
        err_norm = self._sol.func.obj.grad_tol_type
        while not accepted:
            # Obtain current value and gradient.
            val, val_err = self._sol.F
            grad, grad_err = self._sol.G

            # Find step.
            self._step.gradient = grad
            self._step.radius = self._r
            self._step.tolerance = step_tol
            step, _ = self._step.get_step()

            # Calculate projected change.
            prchg = grad(step)
            prchg_err = err_norm.estimated_error(step.measure, grad_err)
            if prchg_err > self._par.margin_proj_desc * abs(prchg):
                bar_tol, grad_tol, step_tol = self._errtol(tau, None)
                self._sol.val_tol = bar_tol
                self._sol.grad_tol = grad_tol
                tau, tau_err = self._sol.instationarity
                continue

            # Calculate functional values at new solution. Reject if non-interior.
            new_sol = BarrierSolution(
                self._sol.func,
                self._sol.x ^ step,
                self._sol.mu,
                bar_tol,
                grad_tol,
                self._sol.eps
            )
            new_val, new_err = new_sol.f
            if numpy.any(new_val[1:] - new_err[1:] <= 0.0):
                # Decrease trust region radius.
                self._r /= 2

                # Terminate if step is too small.
                if self._r < 1000 * numpy.finfo(float).eps:
                    self._status = type(self).Status.SmallStep

                # Increase rejection counter.
                self._stats.n_reject += 1

                # Adjust error tolerances.
                bar_tol, grad_tol, step_tol = self._errtol(tau, None)
                self._sol.val_tol = bar_tol
                self._sol.grad_tol = grad_tol
                tau, tau_err = self._sol.instationarity
                continue

            # Calculate step quality.
            new_val, new_val_err = new_sol.F
            rho = (new_val - val) / prchg
            rho_err = (new_val_err + val_err) / abs(prchg)

            # Update curvature estimate.
            self._updcurv(step.measure, prchg, new_val - val)

            # Decide whether or not to accept.
            if rho - rho_err >= self._par.thres_accept:
                # Accept solution.
                self._sol = new_sol
                accepted = True

                # Update stats.
                self._stats.last_step = step.measure
                self._stats.n_iter += 1

                # Check whether to increase trust region radius.
                if rho >= self._par.thres_tr_expand:
                    self._r = min(
                        self._sol.func.obj.input_space.measure,
                        2 * self._r
                    )
                    logger.debug(f'TR radius increased to {self._r}')
            elif rho + rho_err < self._par.thres_reject:
                # Decrease trust region radius.
                self._r /= 2
                logger.debug(f'rho = {rho}; TR radius decreased to {self._r}')

                # Terminate if step is too small.
                if self._r < 1000 * numpy.finfo(float).eps:
                    self._status = type(self).Status.SmallStep

                # Increase rejection counter.
                self._stats.n_reject += 1

            # Adjust error tolerances.
            bar_tol, grad_tol, step_tol = self._errtol(tau, rho)
            self._sol.val_tol = bar_tol
            self._sol.grad_tol = grad_tol
            tau, tau_err = self._sol.instationarity

    def solve(self) -> None:
        '''
        Run main solver loop.
        '''
        # Print initial solution.
        func_val, _ = self._sol.f
        instat_val, _ = self._sol.instationarity
        self._log.push_line(
            iter=self._stats.n_iter,
            objval=func_val[0],
            instat=instat_val,
            barrier=self._sol.mu,
        )

        # Start main loop.
        iter_start = self._stats.n_iter
        self._status = type(self).Status.Running
        while self._status.is_running and (
            self._par.max_iter is None
            or self._stats.n_iter < iter_start + self._par.max_iter
        ):
            # Remember the number of rejections at the start.
            reject_pre = self._stats.n_reject

            # Perform a step.
            self.step()

            # Print current state.
            func_val, _ = self._sol.f
            instat_val, _ = self._sol.instationarity
            self._log.push_line(
                iter=self._stats.n_iter,
                objval=func_val[0],
                instat=instat_val,
                barrier=self._sol.mu,
                step=self._stats.last_step,
                rejected=self._stats.n_reject - reject_pre
            )

        # Set "maximum iterations exceeded" state.
        if (
            self._status.is_running and self._par.max_iter is not None
            and self._stats.n_iter - iter_start >= self._par.max_iter
        ):
            self._status = type(self).Status.IterationMaximum

        # Print info line.
        print(f'Terminated: {self._status.message}', flush=True)
