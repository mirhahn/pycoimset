'''
Main Lotka-Volterra fishing ODE system.
'''

from typing import Optional, cast, overload

import numpy
from numpy.typing import ArrayLike
import scipy.integrate

from .typing import OdeSolutionLike


__all__ = [
    'LotkaForwardIVP',
    'LotkaForwardErrorIVP',
    'LotkaAdjointIVP',
    'LotkaAdjointErrorIVP',
    'LotkaGradientDensity',
    'LotkaGradientDensityErrorIVP',
]


class LotkaForwardIVP:
    #: Constants determining the impact of the fishing control on the
    #: abundances of the species in the system. Must have shape `(2,)`.
    coefficients: numpy.ndarray

    #: Initial abundance vector. Does not include the quadrature state. Must
    #: have shape `(2,)`.
    _initial: numpy.ndarray

    #: Time range. Tuple of start and end time where start time must be
    #: strictly smaller than end time.
    time_range: tuple[float, float]

    def __init__(self, coeff, initial_state, start_time, end_time):
        self.coefficients = numpy.array(coeff).flatten()
        self._initial = numpy.array(initial_state).flatten()
        self.time_range = (float(start_time), float(end_time))

        if self.coefficients.shape != (2,):
            raise ValueError('`coeff` must have shape (2,).')
        if self._initial.shape != (2,):
            raise ValueError('`initial_state` must have shape (2,).')
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError('`start_time` must be strictly smaller than '
                             '`end_time`.')

    @property
    def initial_state(self):
        return numpy.concatenate((self._initial, [0.0]))

    def __call__(self, _, state, control):
        '''
        Evaluate right hand side of the main ODE system.

        :param state: State variable. Shape must be `(3,)` or `(3, m)` where
                      `m` is the number of time points.
        :type state: array-like
        :param control: Control variable. Must be float or 1-D array-like. If
                        array-like, shape must be `(m,)` where `m` is the
                        number of time points.
        '''
        # Cast relevant array-likes to arrays.
        yval = numpy.asarray(state)
        wval = numpy.asarray(control)
        cval = self.coefficients

        # Return derivative
        return numpy.stack([
            yval[0, ...] * (1 - yval[1, ...] - cval[0] * wval),
            yval[1, ...] * (-1 + yval[0, ...] - cval[1] * wval),
            numpy.sum(numpy.square(yval[:-1, ...] - 1), axis=0)
        ])

    def deriv_state(self, state, control):
        '''
        Generate Jacobian of with respect to the state.

        :param state: State variable. Must have shape `(3,)` or `(3, m)`.
        :type state: array-like
        :param control: Control variable. Must be either a float or an
                        array-like of a shape that is broadcastable to `(m,)`.
        :type control: float or array-like
        :return: Stack of Jacobians. This has shape `(3, 3)`o r `(3, 3, m)`.
        :rtype: numpy.ndarray
        '''
        yval = numpy.asarray(state)
        wval = numpy.asarray(control)
        cval = self.coefficients
        zero = numpy.zeros(yval.shape[1:])

        return numpy.asarray([
            [1 - yval[1, ...] - cval[0] * wval, -yval[0, ...], zero],
            [yval[1, ...], -1 + yval[0, ...] - cval[1] * wval, zero],
            [2 * (yval[0, ...] - 1), 2 * (yval[1, ...] - 1),   zero]
        ]).reshape((yval.shape[0], yval.shape[0], *yval.shape[1:]))

    def deriv_state_state(self, state):
        '''
        Generate second derivatives with respect to state.

        :param state: State variable. Must have shape `(3,)` if `time` is a
                      scalar or `(3, m)`.
        :type state: array-like
        :return: Stack of Hessians. This has shape `(3, 3, 3)` or
                 `(3, 3, 3, m)`.
        :rtype: numpy.ndarray
        '''
        yval = numpy.asarray(state)

        return numpy.broadcast_to(
            numpy.array([
                [
                    [0.0,  -1.0, 0.0],
                    [-1.0,  0.0, 0.0],
                    [0.0,   0.0, 0.0]
                ],
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ],
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]
            ])[:, :, :, *(numpy.newaxis for _ in range(yval.ndim - 1))],
            shape=(3, 3, 3, *yval.shape[1:])
        )

    def deriv_ctrl(self, state):
        '''
        Generate derivatives with respect to control.

        :param state: State variable. Must have shape `(3,)` if `time` is a
                      scalar or `(3, m)`.
        :type state: array-like
        :return: Stack of Jacobian vectors. This has shape `(3,)` if `time` is
                 a scalar or `(3, m)`.
        :rtype: numpy.ndarray
        '''
        yval = numpy.asarray(state)
        cval = self.coefficients
        zeros = numpy.zeros(yval.shape[1:])

        return numpy.stack((
            -cval[0] * yval[0, ...],
            -cval[1] * yval[1, ...],
            zeros
        ))

    def deriv_ctrl_state(self, state):
        '''
        Generate second derivatives of control derivative with respect to
        state.

        :param time: Time variable. This is ignored because the
                     Lotka-Volterra system is not time-dependent. It should be
                     either a float or a 1-D array.
        :type time: float or 1-D array-like
        :param state: State variable. Must have shape `(3,)` if `time` is a
                      scalar or `(3, m)` is `time` is an array of length `m`.
        :type state: array-like
        :param control: Control variable. Must be either a float or an
                        array-like of a shape that is broadcastable to `(m,)`
                        where `m` is the length of `time`.
        :type control: float or array-like
        :return: Stack of Hessian vectors. This has shape `(3, 3)` if `time`
                 is a scalar or `(3, 3, m)` if `time` is an array-like.
        :rtype: numpy.ndarray
        '''
        yval = numpy.asarray(state)
        cval = self.coefficients

        return numpy.broadcast_to(
            numpy.stack((
                (-cval[0],      0.0, 0.0),
                (0.0,      -cval[1], 0.0),
                (0.0,           0.0, 0.0)
            ))[..., *(numpy.newaxis for _ in range(yval.ndim - 1))],
            (3, 3, *yval.shape[1:])
        )


class LotkaForwardErrorIVP:
    '''
    ODE system for primal error estimation.
    '''

    #: Underlying forward IVP.
    fwd_ivp: LotkaForwardIVP

    #: Trajectory for the forward IVP.
    fwd_sol: Optional[scipy.integrate.OdeSolution]

    def __init__(self, fwd_ivp: LotkaForwardIVP):
        self.fwd_ivp = fwd_ivp
        self.fwd_sol = None

    @property
    def time_range(self) -> tuple[float, float]:
        '''
        Return time range of underlying ODE.
        '''
        return self.fwd_ivp.time_range

    @property
    def initial_state(self) -> numpy.ndarray:
        '''
        Initial state.
        '''
        return numpy.zeros((3,))

    def __call__(self, time, state, control):
        # Retrieve trajectory and forward ODE.
        fwd_sol = cast(scipy.integrate.OdeSolution, self.fwd_sol)
        fwd_ode = self.fwd_ivp

        # Convert inputs.
        tval = numpy.asarray(time)
        yval = numpy.asarray(state)
        wval = numpy.asarray(control)

        # Evaluate interpolated solution and time derivative of interpolant.
        y_fwd = fwd_sol(tval)
        ys_fwd = fwd_sol.deriv(tval)    # type: ignore

        # Calculate actual derivative.
        ys_tgt = fwd_ode(tval, y_fwd, wval)
        jac_fwd = fwd_ode.deriv_state(y_fwd, wval)

        # Rearrange axes for matmul.
        yval = numpy.moveaxis(yval, 0, -1)[..., numpy.newaxis]
        jac_fwd = numpy.moveaxis(jac_fwd, (0, 1), (-2, -1))

        # Shift target for objective derivative based on state error.
        ys_tgt -= numpy.moveaxis(jac_fwd @ yval, (-2, -1), (0, 1)).squeeze(1)

        # Calculate residual of state.
        return ys_fwd - ys_tgt


class LotkaAdjointIVP:
    '''
    ODE system for the adjoint state.
    '''

    #: Reference to the primal IVP.
    fwd_ivp: LotkaForwardIVP

    #: Solution trajectory of the primal IVP.
    fwd_sol: Optional[scipy.integrate.OdeSolution]

    def __init__(self, fwd_ivp: LotkaForwardIVP):
        '''
        Default constructor.

        :param fwd_ivp: Primal ODE system to track.
        :type fwd_ivp: :class:`LotkaForwardIVP`
        '''
        self.fwd_ivp = fwd_ivp

    @property
    def time_range(self) -> tuple[float, float]:
        '''
        Time range of the adjoint system.
        '''
        t0, t1 = self.fwd_ivp.time_range
        return (0.0, t1 - t0)

    @property
    def initial_state(self) -> numpy.ndarray:
        '''
        Initial state of the adjoint system.
        '''
        return numpy.array([0.0, 0.0, 1.0])

    @overload
    def import_times(self, times: float) -> float:
        ...

    @overload
    def import_times(self, times: ArrayLike) -> numpy.ndarray:
        ...

    def import_times(self, times):
        '''
        Import a time vector to the reversed time scale.

        This method takes a scalar or array-like of times within the time
        range of the primal IVP and converts them to adjoint system times.
        Its counterpart is `export_times` which converts from the adjoint time
        scale to the original time scale.

        :param times: Time value. Can be scalar or array-like. If this
                      is array-like, it is first converted to a
                      `numpy.ndarray`.
        :type times: float or array-like
        :rtype: float or `numpy.ndarray`
        '''
        if not isinstance(times, float):
            times = numpy.asarray(times)
        _, tend = self.fwd_ivp.time_range
        return tend - times

    @overload
    def export_times(self, times: float) -> float:
        ...

    @overload
    def export_times(self, times: ArrayLike) -> numpy.ndarray:
        ...

    def export_times(self, times):
        '''
        Export a time vector to the original time scale.

        This is the counterpart to `import_times`. It converts times on the
        adjoint time scale back to the original IVP's time scale.

        :param times: Time value. Can be scalar or array-like. If this is
                      array-like, it is first converted to a `numpy.ndarray`.
        :type times: float or array-like
        :rtype: float or `numpy.ndarray`
        '''
        if not isinstance(times, float):
            times = numpy.asarray(times)
        _, tend = self.fwd_ivp.time_range
        return tend - times

    def __call__(self, time, state, control):
        # Convert all relevant inputs into numpy arrays.
        tval = numpy.asarray(time)
        yval = numpy.asarray(state)
        wval = numpy.asarray(control)

        # Convert times to original time scale.
        torig = self.export_times(tval)

        # Retrieve solution trajectory and primal ODE.
        fwd_ivp = self.fwd_ivp
        fwd_sol = cast(scipy.integrate.OdeSolution, self.fwd_sol)

        # Evaluate interpolated solution.
        y_fwd = fwd_sol(torig)
        jac_fwd = fwd_ivp.deriv_state(y_fwd, wval)

        # Rearrange dimensions for matmul.
        yval = numpy.moveaxis(yval[:, numpy.newaxis, ...], (0, 1), (-2, -1))
        jac_fwd = numpy.moveaxis(jac_fwd, (0, 1), (-1, -2))

        # Calculate matrix product
        return numpy.moveaxis(jac_fwd @ yval, (-2, -1), (0, 1)).squeeze(1)


class LotkaAdjointErrorIVP:
    '''
    Initial value problem for adjoint state error estimation.
    '''

    adj_ivp: LotkaAdjointIVP

    fwd_sol: Optional[scipy.integrate.OdeSolution]

    fwd_err_sol: Optional[scipy.integrate.OdeSolution]

    adj_sol: Optional[scipy.integrate.OdeSolution]

    def __init__(self, adj_ivp: LotkaAdjointIVP):
        '''
        Create a new adjoint state error estimation system.

        :param adj_ivp: Underlying adjoint IVP.
        :type adj_ivp: :class:`LotkaAdjointIVP`
        '''
        self.adj_ivp = adj_ivp
        self.fwd_sol = None
        self.fwd_err_sol = None
        self.adj_sol = None

    @property
    def fwd_ivp(self) -> LotkaForwardIVP:
        '''
        Underlying forward IVP.
        '''
        return self.adj_ivp.fwd_ivp

    @property
    def time_range(self) -> tuple[float, float]:
        '''
        Time range of the adjoint state error estimation IVP.

        The error estimation IVP operates on the same time scale as the
        adjoint IVP.
        '''
        return self.adj_ivp.time_range

    @property
    def initial_state(self) -> numpy.ndarray:
        '''
        Initial state for the adjoint state error estimation IVP.
        '''
        return numpy.zeros((3,))

    def __call__(self, time, state, control):
        # Obtain references to underlying IVP solutions.
        fwd_ode = self.fwd_ivp
        adj_ivp = self.adj_ivp
        fwd_sol = cast(scipy.integrate.OdeSolution, self.fwd_sol)
        fwd_err_sol = cast(scipy.integrate.OdeSolution, self.fwd_err_sol)
        adj_sol = cast(scipy.integrate.OdeSolution, self.adj_sol)

        # Cast all relevant data into NumPy arrays.
        tval = numpy.asarray(time)
        yval = numpy.asarray(state)
        wval = numpy.asarray(control)
        torig = adj_ivp.import_times(tval)

        # Evaluate interpolated solution with error adjustment.
        y_fwd = fwd_sol(torig)
        jac_fwd = fwd_ode.deriv_state(y_fwd, wval)
        hess_fwd = fwd_ode.deriv_state_state(y_fwd)
        err_fwd = fwd_err_sol(torig)
        y_adj = adj_sol(tval)
        ys_adj = adj_sol.deriv(tval)    # type: ignore

        # Rearrange dimensions for Jacobian adjustment.
        n_y = y_fwd.shape[0]
        hess_fwd = numpy.moveaxis(
            hess_fwd, (0, 1, 2), (-3, -2, -1)
        )[..., numpy.newaxis]
        err_fwd = numpy.broadcast_to(
            numpy.moveaxis(
                err_fwd[numpy.newaxis, numpy.newaxis, numpy.newaxis, ...],
                (0, 1, 2, 3),
                (-4, -3, -2, -1)
            ),
            (*yval.shape[1:], n_y, n_y, 1, n_y)
        )

        # Adjust Jacobian.
        jac_fwd += numpy.moveaxis(
            err_fwd @ hess_fwd,
            (-4, -3, -2, -1),
            (0, 1, 2, 3)
        ).squeeze((2, 3))

        # Rearrange dimensions for target calculation.
        y_adj = numpy.moveaxis(y_adj - yval, 0, -1)[..., numpy.newaxis, :]
        jac_fwd = numpy.moveaxis(jac_fwd, (0, 1), (-2, -1))

        # Calculate target for time derivative of y_adj.
        ys_tgt = numpy.moveaxis((y_adj @ jac_fwd).squeeze(-2), -1, 0)

        # Calculate residual
        return ys_adj - ys_tgt


class LotkaGradientDensity:
    '''
    Unsigned gradient density function.
    '''

    #: Underlying adjoint IVP.
    adj_ivp: LotkaAdjointIVP

    #: Primal solution.
    fwd_sol: Optional[scipy.integrate.OdeSolution]

    #: Primal error estimate.
    fwd_err_sol: Optional[scipy.integrate.OdeSolution]

    #: Adjoint solution.
    adj_sol: Optional[scipy.integrate.OdeSolution]

    #: Adjoint error estimate.
    adj_err_sol: Optional[scipy.integrate.OdeSolution]

    def __init__(self, adj_ivp: LotkaAdjointIVP):
        '''
        Create a new gradient density function evaluator.

        :param adj_ivp: Adjoint IVP for function evaluation.
        :type adj_ivp: :class:`LotkaAdjointIVP`
        '''
        self.adj_ivp = adj_ivp

    @property
    def fwd_ivp(self) -> LotkaForwardIVP:
        '''
        Underlying forward IVP.
        '''
        return self.adj_ivp.fwd_ivp

    def __call__(self, time):
        '''
        Evaluate unsigned gradient density function.

        Note that this does not include the sign flip based on the value of
        `w`. You have to perform this sign flip afterwards.

        :param time: Time variable. Must have shape `(,)` or `(n),`
        :type time: array-like
        :param w: Control variable.
        :type w: array-like of shape `(,)` or `(n,)`

        :rtype: `numpy.ndarray`, shape `(,)` or `(m,)`
        '''
        # Obtain references to relevant trajectories and IVPs.
        fwd_ivp = self.fwd_ivp
        adj_ivp = self.adj_ivp
        fwd_sol = cast(scipy.integrate.OdeSolution, self.fwd_sol)
        adj_sol = cast(scipy.integrate.OdeSolution, self.adj_sol)

        # Obtain time arrays.
        tval = numpy.asarray(time)
        tadj = adj_ivp.import_times(tval)

        # Evaluate primal and adjoint solution.
        yval = fwd_sol(tval)
        zval = adj_sol(tadj)

        # Calculate derivative of ODE rhs with respect to control
        jac_ctrl = fwd_ivp.deriv_ctrl(yval)

        # Rearrange dimensions for matrix multiplication.
        zval = numpy.moveaxis(zval, 0, -1)[..., numpy.newaxis, :]
        jac_ctrl = numpy.moveaxis(jac_ctrl, 0, -1)[..., numpy.newaxis]

        # Calculate gradient density.
        return numpy.squeeze(zval @ jac_ctrl, (-2, -1))

    def deriv(self, time):
        '''
        Evaluate time derivative of unsigned gradient density function.

        Note that this does not include the sign flip based on the value of
        `w`. You have to perform this sign flip afterwards.

        :param time: Time variable. Must have shape `(,)` or `(n),`
        :type time: array-like
        :param w: Control variable.
        :type w: array-like of shape `(,)` or `(n,)`

        :rtype: `numpy.ndarray`, shape `(,)` or `(m,)`
        '''
        # Obtain references to relevant trajectories and IVPs.
        fwd_ivp = self.fwd_ivp
        adj_ivp = self.adj_ivp
        fwd_sol = cast(scipy.integrate.OdeSolution, self.fwd_sol)
        adj_sol = cast(scipy.integrate.OdeSolution, self.adj_sol)

        # Obtain time arrays.
        tval = numpy.asarray(time)
        tadj = adj_ivp.import_times(tval)

        # Compute relevant inputs.
        yval = fwd_sol(tval)
        ysval = fwd_sol.deriv(tval)     # type: ignore
        zval = adj_sol(tadj)
        zsval = -adj_sol.deriv(tadj)    # type: ignore

        # Calculate required derivatives.
        f_w = fwd_ivp.deriv_ctrl(yval)
        f_wy = fwd_ivp.deriv_ctrl_state(yval)

        # Rearrange dimensions for matrix multiplication.
        zval = numpy.moveaxis(zval, 0, -1)[..., numpy.newaxis, :]
        zsval = numpy.moveaxis(zsval, 0, -1)[..., numpy.newaxis, :]
        ysval = numpy.moveaxis(ysval, 0, -1)[..., numpy.newaxis]
        f_w = numpy.moveaxis(f_w, 0, -1)[..., numpy.newaxis]
        f_wy = numpy.moveaxis(f_wy, (0, 1), (-2, -1))

        # Calculate derivative.
        return numpy.squeeze(zsval @ f_w + zval @ f_wy @ ysval, (-2, -1))

    def deriv_exact(self, time, control):
        '''
        Approximate the derivative of the exact gradient density function.

        This function is used for a posteriori error estimation. It
        approximates what the exact derivative of the gradient density should
        be. It takes into account estimates of primal solution and adjoint
        solution errors by adding first-order corrections to all quantities
        that depend on these inputs.
        '''
        # Get references to IVPs.
        fwd_ivp = self.fwd_ivp
        adj_ivp = self.adj_ivp

        # Convert time and control into vectors.
        tval = numpy.asarray(time)
        wval = numpy.asarray(control)
        tadj = adj_ivp.import_times(tval)

        # Interpolate primal state, primal error, adjoint state, and adjoint
        # error.
        yval = cast(scipy.integrate.OdeSolution, self.fwd_sol)(tval)
        yerr = cast(scipy.integrate.OdeSolution, self.fwd_err_sol)(tval)
        zval = cast(scipy.integrate.OdeSolution, self.adj_sol)(tadj)
        zerr = cast(scipy.integrate.OdeSolution, self.adj_err_sol)(tadj)

        # Calculate necessary derivatives of ODE right hand side.
        rhs_val = fwd_ivp(tval, yval, wval)
        rhs_der_y = fwd_ivp.deriv_state(yval, wval)
        rhs_der_yy = fwd_ivp.deriv_state_state(yval)
        rhs_der_w = fwd_ivp.deriv_ctrl(yval)
        rhs_der_wy = fwd_ivp.deriv_ctrl_state(yval)

        # Rearrange dimensions for matrix multiplication.
        zval = numpy.moveaxis(zval, 0, -1)[..., numpy.newaxis, :]
        zerr = numpy.moveaxis(zerr, 0, -1)[..., numpy.newaxis, :]
        yerr = numpy.moveaxis(yerr, 0, -1)[..., numpy.newaxis]
        rhs_val = numpy.moveaxis(rhs_val, 0, -1)[..., numpy.newaxis]
        rhs_der_y = numpy.moveaxis(rhs_der_y, (0, 1), (-2, -1))
        rhs_der_yy = numpy.moveaxis(rhs_der_yy, (0, 1, 2), (-3, -1, -2))
        rhs_der_w = numpy.moveaxis(rhs_der_w, 0, -1)[..., numpy.newaxis]
        rhs_der_wy = numpy.moveaxis(rhs_der_wy, (0, 1), (-2, -1))

        # Perform error based adjustments. Note that all third derivatives are
        # zero and therefore, we need not adjust second derivatives.
        zval -= zerr
        rhs_val -= rhs_der_y @ yerr
        rhs_der_y -= numpy.squeeze(
            rhs_der_yy @ yerr[..., numpy.newaxis, :, :], -1
        )
        rhs_der_w -= rhs_der_wy @ yerr

        # Perform matrix multiplications to approximate exact derivative of
        # gradient density.
        return numpy.squeeze(
            zval @ (rhs_der_wy @ rhs_val - rhs_der_y @ rhs_der_w), (-2, -1)
        )


class LotkaGradientDensityErrorIVP:
    '''
    Initial value problem for gradient density error estimation.
    '''

    #: Gradient density evaluator.
    grad_dens: LotkaGradientDensity

    #: Gradient density trajectory.
    grad_dens_traj: Optional[OdeSolutionLike]

    def __init__(self, grad_dens: LotkaGradientDensity):
        '''
        Initialize a new initial value problem.
        '''
        self.grad_dens = grad_dens
        self.grad_dens_traj = None

    @property
    def adj_ivp(self) -> LotkaAdjointIVP:
        '''Associated adjoint initial value problem.'''
        return self.grad_dens.adj_ivp

    @property
    def fwd_ivp(self) -> LotkaForwardIVP:
        '''Associated primal initial value problem.'''
        return self.grad_dens.fwd_ivp

    @property
    def time_range(self) -> tuple[float, float]:
        '''Time range of initial value problem.'''
        return self.fwd_ivp.time_range

    @property
    def initial_state(self) -> numpy.ndarray:
        '''Initial state of initial value problem.'''
        return numpy.zeros((1,))

    def __call__(self, time, _, control):
        '''Evaluate right hand side of error estimation system.'''
        # Convert time and control into NumPy arrays.
        tval = numpy.asarray(time)
        wval = numpy.asarray(control)

        # Evaluate interpolant derivative and target derivatives.
        der_val = self.grad_dens_traj.deriv(tval)       # type: ignore
        der_tgt = self.grad_dens.deriv_exact(tval, wval)

        # Calculate residual.
        return der_val - der_tgt
