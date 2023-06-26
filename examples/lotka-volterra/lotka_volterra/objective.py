# PyCoimset - Python library for set-valued optimization
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

from dataclasses import dataclass, field
import math
from types import NoneType
from typing import Optional

import numpy
from pycoimset.typing import ErrorNorm, Functional, SimilarityClass
from scipy.integrate import OdeSolution, solve_ivp

from .ode import (
    LotkaAdjointErrorIVP,
    LotkaAdjointIVP,
    LotkaForwardErrorIVP,
    LotkaForwardIVP,
    LotkaGradientDensity,
    LotkaGradientDensityErrorIVP,
)
from .polyfit import (
    PolynomialTrajectory,
    merge_time_grids,
    midpoint_time_grid,
    polyfit_quartic,
)
from .space import (
    IntervalSimilarityClass,
    IntervalSimilaritySpace,
    PolynomialSignedMeasure,
)         # pyright: ignore # noqa

__all__ = ['LotkaObjectiveFunctional']


class LotkaObjectiveFunctional(Functional[IntervalSimilaritySpace]):
    '''
    Objective functional for the Lotka-Volterra fishing problem.
    '''

    @dataclass(frozen=True, slots=True)
    class LotkaObjects:
        #: Forward IVP.
        fwd: LotkaForwardIVP

        #: Forward error IVP.
        fwd_err: LotkaForwardErrorIVP = field(init=False)

        #: Adjoint IVP.
        adj: LotkaAdjointIVP = field(init=False)

        #: Adjoint error IVP.
        adj_err: LotkaAdjointErrorIVP = field(init=False)

        #: Gradient density evaluator.
        grad_dens: LotkaGradientDensity = field(init=False)

        #: Gradient density error IVP.
        grad_dens_err: LotkaGradientDensityErrorIVP = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, 'fwd_err',
                               LotkaForwardErrorIVP(self.fwd))
            object.__setattr__(self, 'adj',
                               LotkaAdjointIVP(self.fwd))
            object.__setattr__(self, 'adj_err',
                               LotkaAdjointErrorIVP(self.adj))
            object.__setattr__(self, 'grad_dens',
                               LotkaGradientDensity(self.adj))
            object.__setattr__(self, 'grad_dens_err',
                               LotkaGradientDensityErrorIVP(self.grad_dens))

    @dataclass(slots=True)
    class Trajectories:
        #: Control values.
        ctrl: Optional[list[float]] = None

        #: Forward solution trajectory.
        fwd: Optional[list[OdeSolution]] = None

        # Forward error trajectory.
        fwd_err: Optional[list[OdeSolution]] = None

        # Adjoint solution trajectory.
        adj: Optional[list[OdeSolution]] = None

        #: Adjoint error solution.
        adj_err: Optional[list[OdeSolution]] = None

        #: Unsigned gradient density.
        grad_dens: Optional[list[PolynomialTrajectory]] = None

        #: Gradient density error solution.
        grad_dens_err: Optional[list[OdeSolution]] = None

        def clear_grad(self) -> None:
            '''Clear all trajectories related to gradient error.'''
            self.adj = None
            self.adj_err = None
            self.grad_dens = None
            self.grad_dens_err = None

        def clear(self) -> None:
            '''Clear all trajectories.'''
            self.ctrl = None
            self.fwd = None
            self.fwd_err = None
            self.clear_grad()

    #: Input measure space. This determines start and end times.
    _space: IntervalSimilaritySpace

    #: Argument similarity class.
    _arg: Optional[IntervalSimilarityClass]

    #: Value tolerance.
    _valtol: float

    #: Gradient tolerance.
    _gradtol: float

    #: Absolute integration tolerance.
    _abstol: Optional[float]

    #: Relative integration tolerance.
    _reltol: Optional[float]

    #: Result cache for `get_value()`.
    _val_result: Optional[tuple[float, float]]

    #: Result cache for `get_gradient()`.
    _grad_result: Optional[tuple[PolynomialSignedMeasure, float]]

    #: IVP objects.
    _ivp: LotkaObjects

    #: Trajectory objects.
    _traj: Trajectories

    def __init__(self, space: IntervalSimilaritySpace):
        self._space = space
        self._arg = None
        self._valtol = 1e-3
        self._gradtol = 1e-3
        self._abstol = None
        self._reltol = None
        self._val_result = None
        self._grad_result = None
        self._ivp = LotkaObjectiveFunctional.LotkaObjects(
            LotkaForwardIVP([0.4, 0.2], [0.5, 0.7], *space.time_range)
        )
        self._traj = LotkaObjectiveFunctional.Trajectories()

    @property
    def input_space(self) -> IntervalSimilaritySpace:
        return self._space

    @property
    def arg(self) -> Optional[IntervalSimilarityClass]:
        return self._arg

    @arg.setter
    def arg(self, arg: Optional[SimilarityClass]) -> None:
        # Skip if identical to previous argument.
        if arg == self._arg:
            return

        # Raise exception if type is incompatible.
        if not isinstance(arg, IntervalSimilarityClass):
            raise TypeError('`arg` must be an instance of '
                            '`IntervalSimilarityClass`')

        # Set argument.
        self._arg = arg

        # Invalidate all cached outputs and tolerances.
        self._abstol = None
        self._reltol = None
        self._val_result = None
        self._grad_result = None
        self._traj.clear()

    @property
    def val_tol(self) -> float:
        return self._valtol

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        if tol <= 0.0:
            raise ValueError('tolerance must be strictly positive')

        self._valtol = tol

        if self._val_result is not None and self._val_result[1] > tol:
            self._abstol = None
            self._reltol = None
            self._grad_result = None
            self._val_result = None
            self._traj.clear()

    @property
    def grad_tol(self) -> float:
        return self._gradtol

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        if tol <= 0.0:
            raise ValueError('tolerance must be strictly positive')

        self._gradtol = tol

        if self._grad_result is not None and self._grad_result[1] > tol:
            self._grad_result = None
            self._traj.clear_grad()

    @property
    def grad_tol_type(self) -> ErrorNorm:
        return ErrorNorm.Linfty

    @property
    def ivp_objects(self) -> LotkaObjects:
        return self._ivp

    @property
    def trajectories(self) -> Trajectories:
        return self._traj

    @property
    def _inttol(self) -> tuple[float, float]:
        '''Integration tolerances for IVP solvers.'''
        if self._abstol is None or self._reltol is None:
            t0, tf = self._ivp.fwd.time_range
            dt = tf - t0
            self._reltol = float(100 * numpy.finfo(float).eps)
            self._abstol = min(self.val_tol / dt, self.grad_tol / (2 * dt), 1e-6)
        return self._abstol, self._reltol

    @_inttol.setter
    def _inttol(self, tol: tuple[float, float] | NoneType) -> None:
        if tol is None:
            self._abstol = None
            self._reltol = None
            return

        if tol[0] <= 0.0 or tol[1] < 0.0:
            raise ValueError('integration tolerances must be positive')
        self._abstol, self._reltol = tol

    def get_value(self) -> tuple[float, float]:
        # Test if cache is available.
        if self._val_result is not None:
            return self._val_result

        # Raise a `ValueError` if `arg` is not set.
        if self._arg is None:
            raise ValueError('`arg` is not set')

        # Get start, end, and switching times.
        _, tf = self._ivp.fwd.time_range
        ts = self._arg.switch_times

        # Get integration tolerances.
        atol, rtol = self._inttol

        # Perform forward simulation until tolerances are satisfied.
        val = math.nan
        val_err = math.inf
        traj_ctrl: list[float] = []
        traj_fwd: list[OdeSolution] = []
        traj_err: list[OdeSolution] = []
        while True:
            # Running variables and trajectory list for forward IVP.
            t0, _ = self._ivp.fwd.time_range
            x0 = numpy.array(self._ivp.fwd.initial_state)
            e0 = numpy.array(self._ivp.fwd_err.initial_state)
            w = 0.0

            # Iterate over switching times
            for t1 in numpy.concatenate((ts, [tf])):
                # Short-circuit if switching time is in the past.
                if t1 <= t0:
                    w = 1 - w
                    continue

                # Solve main initial value problem.
                res = solve_ivp(self._ivp.fwd, (t0, t1), x0, method='RK45',
                                dense_output=True, args=(w,),
                                rtol=rtol, atol=atol)

                # Handle integration errors.
                if not res.success:
                    raise RuntimeError('forward integration failure: '
                                       f'{res.message}')

                # Set up error system.
                self._ivp.fwd_err.fwd_sol = res.sol

                # Solve error estimation initial value problem.
                res_err = solve_ivp(self._ivp.fwd_err, (t0, t1), e0,
                                    method='RK45', dense_output=True,
                                    args=(w,), rtol=rtol, atol=atol)

                # Handle integration errors.
                if not res_err.success:
                    raise RuntimeError('forward error estimation failure: '
                                       f'{res_err.message}')

                # Record trajectory snippets.
                traj_ctrl.append(w)
                traj_fwd.append(res.sol)
                traj_err.append(res_err.sol)

                # Advance state.
                x0 = res.y[:, -1]
                e0 = res_err.y[:, -1]
                t0 = t1
                w = 1 - w

            # Determine objective.
            val = x0[2]
            val_err = 2 * abs(e0[2])

            # Terminate loop if error tolerance is satisfied.
            if val_err <= self._valtol:
                break

            # Reduce integration tolerances.
            atol *= self._valtol / (2 * val_err)
            self._inttol = (atol, rtol)

            traj_ctrl = []
            traj_fwd = []
            traj_err = []

        # Once tolerances are satisfied, store the recorded trajectories.
        self._traj.ctrl = traj_ctrl
        self._traj.fwd = traj_fwd
        self._traj.fwd_err = traj_err

        # Return result.
        self._val_result = (val, val_err)
        return self._val_result

    def get_gradient(self) -> tuple[PolynomialSignedMeasure, float]:
        # Use cached result if available.
        if self._grad_result is not None:
            return self._grad_result

        # Raise a `ValueError` if `arg` is not set.
        if self._arg is None:
            raise ValueError('`arg` is not set.')

        # Ensure that the functional is evaluated.
        self.get_value()

        # Retrieve integration tolerances.
        atol, rtol = self._inttol

        # Perform adjoint simulation and gradient generation until tolerance
        # is satisfied.
        traj_adj: list[OdeSolution] = []
        traj_adj_err: list[OdeSolution] = []
        traj_grad: list[PolynomialTrajectory] = []
        traj_grad_err: list[OdeSolution] = []

        while True:
            # DEBUG: Assert presence of forward trajectories.
            assert self._traj.ctrl is not None
            assert self._traj.fwd is not None
            assert self._traj.fwd_err is not None

            # Set initial state for adjoint systems.
            y0 = self._ivp.adj.initial_state
            e0 = self._ivp.adj_err.initial_state

            # Solve adjoint system patch-by-patch.
            for w, sol_fwd, sol_fwd_err in zip(reversed(self._traj.ctrl),
                                               reversed(self._traj.fwd),
                                               reversed(self._traj.fwd_err)):
                # Get start and end times for this control interval.
                tf, t0 = self._ivp.adj.import_times([sol_fwd.t_min,
                                                     sol_fwd.t_max])

                # Configure adjoint IVP.
                self._ivp.adj.fwd_sol = sol_fwd

                # Solve IVP for this patch.
                res_adj = solve_ivp(self._ivp.adj, (t0, tf), y0,
                                    method='RK45', dense_output=True,
                                    args=(w,), rtol=rtol, atol=atol)

                # Handle integration error.
                if not res_adj.success:
                    raise RuntimeError('adjoint integration error: '
                                       f'{res_adj.message}')

                # Configure error estimation IVP.
                self._ivp.adj_err.fwd_sol = sol_fwd
                self._ivp.adj_err.fwd_err_sol = sol_fwd_err
                self._ivp.adj_err.adj_sol = res_adj.sol

                # Solve error estimation IVP for this patch.
                res_adj_err = solve_ivp(self._ivp.adj_err, (t0, tf), e0,
                                        method='RK45', dense_output=True,
                                        args=(w,), rtol=rtol, atol=atol)

                # Handle integration error.
                if not res_adj_err.success:
                    raise RuntimeError('adjoint error estimation error: '
                                       f'{res_adj_err.message}')

                # Save trajectory patches.
                traj_adj.append(res_adj.sol)
                traj_adj_err.append(res_adj_err.sol)

                # Advance state.
                y0 = res_adj.y[..., -1]
                e0 = res_adj_err.y[..., -1]

            # Set up initial state for the gradient density error estimation
            # system.
            e0 = self._ivp.grad_dens_err.initial_state

            # Find gradient density patch-by-patch.
            grad_err = 0.0
            for w, sol_fwd, sol_fwd_err, sol_adj, sol_adj_err in zip(
                self._traj.ctrl,
                self._traj.fwd,
                self._traj.fwd_err,
                reversed(traj_adj),
                reversed(traj_adj_err)
            ):
                # Find joint time grid and midpoint time grid.
                ts = merge_time_grids(
                    sol_fwd.ts,
                    sol_fwd_err.ts,
                    numpy.flip(self._ivp.adj.export_times(sol_adj.ts)),
                    numpy.flip(self._ivp.adj.export_times(sol_adj_err.ts))
                )
                ts_mid = midpoint_time_grid(ts)

                # Configure gradient density evaluator.
                self._ivp.grad_dens.fwd_sol = sol_fwd
                self._ivp.grad_dens.fwd_err_sol = sol_fwd_err
                self._ivp.grad_dens.adj_sol = sol_adj
                self._ivp.grad_dens.adj_err_sol = sol_adj_err

                # Evaluate gradient density at midpoint time grid and
                # its derivative at the original grid points.
                g = self._ivp.grad_dens(ts_mid)
                gs = self._ivp.grad_dens.deriv(ts)

                # Fit a quartic to obtain the (unsigned) gradient patch.
                sol_grad = polyfit_quartic(ts_mid, g, gs)

                # Configure error estimation system for this patch.
                self._ivp.grad_dens_err.grad_dens_traj = sol_grad

                # Solve error estimation system.
                res_grad_err = solve_ivp(self._ivp.grad_dens_err,
                                         (ts[0], ts[-1]), e0,
                                         method='RK45',
                                         dense_output=True,
                                         args=(w,), rtol=rtol, atol=atol)

                # Handle integration errors.
                if not res_grad_err.success:
                    raise RuntimeError('gradient error estimation error: '
                                       f'{res_grad_err.message}')

                # Update gradient density error estimate.
                grad_err = max(grad_err, 2 * numpy.max(
                    numpy.abs(res_grad_err.y)
                ))

                # Otherwise, extend trajectories and advance state.
                traj_grad.append(sol_grad)
                traj_grad_err.append(res_grad_err.sol)

                e0 = res_grad_err.y[..., -1]

            # Terminate loop if gradient error is sufficiently low
            if grad_err <= self._gradtol:
                break

            # Otherwise, reset trajectories, adjust integration tolerances,
            # force re-evaluation of objective value.
            traj_adj = []
            traj_adj_err = []
            traj_grad = []
            traj_grad_err = []

            atol *= self._gradtol / (2 * grad_err)
            self._inttol = (atol, rtol)

            self._val_result = None
            self.get_value()
            atol, rtol = self._inttol

        # Once done, store trajectories.
        self._traj.adj = traj_adj
        self._traj.adj_err = traj_adj_err
        self._traj.grad_dens = traj_grad
        self._traj.grad_dens_err = traj_grad_err

        # Generate signed measure from gradient density trajectory.
        poly = [
            (1 - 2 * w) * sol
            for w, sol in zip(self._traj.ctrl, self._traj.grad_dens)
        ]
        time = numpy.concatenate([poly[0].ts] + [p.ts[1:] for p in poly[1:]])
        coef = numpy.concatenate([p.coef.squeeze(1) for p in poly])
        scale = numpy.concatenate([p._scale for p in poly])
        self._grad_result = (
            PolynomialSignedMeasure(
                self._space,
                PolynomialTrajectory(time, coef, mapscale=scale)
            ),
            grad_err
        )

        return self._grad_result
