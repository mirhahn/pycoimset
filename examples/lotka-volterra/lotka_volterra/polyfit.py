'''
Helpers for polynomial fitting of 'fake' ODE trajectories.
'''

from itertools import repeat
from typing import Optional, cast
from types import NotImplementedType

import numpy
from numpy.typing import ArrayLike, NDArray
import scipy
import sortednp

from .typing import OdeSolutionLike


__all__ = [
    'PolynomialTrajectory',
    'merge_time_grids',
    'midpoint_time_grid',
    'polyfit_quartic',
]


class PolynomialTrajectory(OdeSolutionLike):
    '''
    Facsimile of `scipy.integrate.OdeSolution` that uses vectorized
    polynomial operations.
    '''

    #: Time instants between which local interpolants are defined.
    #: Must be strictly increasing.
    ts: NDArray

    #: Array of coefficients.
    coef: NDArray

    #: Joint window used by all polynomials. Defaults to `(-1.0, 1.0)`.
    window: tuple[float, float]

    #: Mapping parameter.
    _scale: NDArray

    #: Derivative cache.
    _deriv: Optional['PolynomialTrajectory']

    #: Antiderivative cache.
    _antideriv: Optional['PolynomialTrajectory']

    def __init__(self, ts: ArrayLike, coef: ArrayLike,
                 window: Optional[tuple[float, float]] = None,
                 mapscale: Optional[ArrayLike] = None):
        self.ts = numpy.asarray(ts)
        self.coef = numpy.asarray(coef)

        if window is None:
            self.window = (-1.0, 1.0)
        else:
            self.window = window

        # Calculate mapping parameters.
        if mapscale is None:
            x0, x1 = self.window
            self._scale = (x1 - x0) / (self.ts[1:] - self.ts[:-1])
        else:
            self._scale = numpy.asarray(mapscale)

        # Set up cache for derivatives and antiderivatives.
        self._deriv = None
        self._antideriv = None

    def __repr__(self) -> str:
        '''Create string representation.'''
        return (f'PolynomialTrajectory({repr(self.ts)}, {repr(self.coef)}, '
                f'window={repr(self.window)})')

    def __call__(self, t: ArrayLike) -> NDArray:
        '''Evaluate trajectory.'''
        # Convert `t` into a NumPy array.
        t = numpy.asarray(t)
        ts = numpy.atleast_1d(t)
        if ts.ndim > 1:
            raise ValueError('`t` cannot have more than one dimension')

        # Find interpolant indices.
        i = numpy.fmax(
            numpy.searchsorted(
                self.ts[:-1], ts, side='right'
            ) - 1,
            0
        )

        # Convert to local parameters
        x = self._scale[i] * (ts - self.ts[i]) + self.window[0]

        # Broadcast local parameters for matrix multiplication.
        x = x[:, *repeat(numpy.newaxis, self.coef.ndim - 1)]
        x = numpy.broadcast_to(x, (*x.shape[:-1], self.coef.shape[-1] - 1))
        x = numpy.cumprod(x, axis=-1)
        x = x[..., numpy.newaxis]

        # Evaluate polynomial values.
        coef = self.coef[i, ..., numpy.newaxis, :]
        f = coef[..., 0] + (coef[..., 1:] @ x).squeeze(-1)
        f = f.squeeze(-1)

        if t.ndim == 0:
            f.squeeze(0)

        return f

    def piece_eval(self, ts: ArrayLike) -> NDArray:
        '''
        Evaluate individual pieces at a number of points simultaneously.

        :param ts: Time matrix with exactly as many rows as there are
                   polynomials in the trajectory. Each polynomial is
                   evaluated at the times in its corresponding row.
        :type ts: 2-D array-like

        :return: An array of evaluated polynomial values. The first
                 dimension corresponds to the polynomials, the second
                 to the evaluation times.
        :rtype: `numpy.ndarray` of shape `(n_p, n_t, *)`
        '''
        # Convert time array if necessary.
        ts = numpy.atleast_2d(numpy.asarray(ts))
        if ts.ndim > 2:
            raise ValueError('`ts` must be at most a 2-D array')

        # Build the cumulative product matrix.
        n_deg = self.coef.shape[-1] - 1
        xs = (
            self._scale[:, numpy.newaxis] * (ts - self.ts[:-1, numpy.newaxis])
            + self.window[0]
        )
        xs = xs[..., *repeat(numpy.newaxis, self.coef.ndim - 1)]
        xs = numpy.broadcast_to(xs, (*xs.shape[:-1], n_deg))
        xs = numpy.cumprod(xs, axis=-1)

        # Compute the output values.
        coef = self.coef[:, numpy.newaxis, ...]
        fs = coef[..., 0] + (coef[..., numpy.newaxis, 1:]
                             @ xs[..., numpy.newaxis]).squeeze((-2, -1))

        return fs

    def poly_deriv(self, n: int = 1) -> 'PolynomialTrajectory':
        '''Return derivative as a piecewise polynomial.'''
        if n < 0:
            raise ValueError('`n` must be non-negative.')
        if n == 0:
            return self
        if n > 1:
            return self.poly_deriv().poly_deriv(n - 1)

        # Return cached result if available.
        if self._deriv is not None:
            return self._deriv

        # Get number of coefficients and short-circuit if constant polynomial
        # should be returned.
        n_coef = self.coef.shape[-1]
        if n_coef < n:
            poly = PolynomialTrajectory(
                [self.t_min, self.t_max],
                numpy.zeros((1, *self.coef.shape[1:-1], 1))
            )
            self._deriv = poly
            return poly

        # Calculate coefficients.
        coef = (
            self.coef[..., 1:]
            * numpy.arange(1, n_coef)[*repeat(numpy.newaxis,
                                              self.coef.ndim - 1), :]
            * self._scale[:, *repeat(numpy.newaxis, self.coef.ndim - 1)]
        )

        # Create new polynomial trajectory and cache result.
        # Note: We could set `self` to be `_antideriv` of `poly`, but that
        #       would cause circular references.
        poly = PolynomialTrajectory(self.ts, coef, window=self.window,
                                    mapscale=self._scale)
        self._deriv = poly

        return poly

    def deriv(self, t: ArrayLike) -> NDArray:
        '''Evaluate first derivative.'''
        return self.poly_deriv()(t)

    def poly_anti_deriv(self, n: int = 1) -> 'PolynomialTrajectory':
        '''Return antiderivative as a piecewise polynomial.'''
        if n < 0:
            raise ValueError('`n` must be non-negative.')
        if n == 0:
            return self
        if n > 1:
            return self.poly_anti_deriv().poly_anti_deriv(n - 1)

        # Return cached result if available.
        if self._antideriv is not None:
            return self._antideriv

        # Get number of coefficients.
        n_coef = self.coef.shape[-1]

        # Calculate coefficients.
        coef = numpy.empty((*self.coef.shape[:-1], self.coef.shape[-1] + 1))
        coef[..., 1:] = self.coef / numpy.arange(1, n_coef + 1)[
            *(numpy.newaxis for _ in self.coef.shape[:-1]), :
        ] / self._scale[:, *repeat(numpy.newaxis, self.coef.ndim - 1)]

        # Calculate constant shift by cumulative sum.
        _, x = self.window
        if x in (0.0, 1.0):
            x = numpy.broadcast_to(x, (n_coef,))
        else:
            x = numpy.cumprod(numpy.broadcast_to(x, (n_coef,)))
        x = x[*(numpy.newaxis for _ in coef.shape[:-2]), :, numpy.newaxis]
        coef[0, ..., 0] = 0.0
        coef[1:, ..., 0] = numpy.cumsum(coef[1:, ..., numpy.newaxis, 1:]
                                        @ x, axis=0).squeeze((-2, -1))

        # Create new polynomial trajectory and cache result.
        poly = PolynomialTrajectory(self.ts, coef, window=self.window,
                                    mapscale=self._scale)
        self._antideriv = poly

        return poly

    def roots(self, fill: float | complex = numpy.nan) -> NDArray:
        '''
        Real roots of each polynomial.

        This only works for polynomials up to degree 4 and uses
        Ferrari's method with an added Newton step for refinement. The
        output array always has shape `(..., d)` where `...` is the
        shape of the polynomial array and `d` is the maximal degree of
        a polynomial in this trajectory.

        Since not every polynomial has `d` real roots, the remaining
        roots are replaced with `numpy.inf`.
        '''
        # Construct an empty output array.
        root = numpy.empty((*self.coef.shape[:-1], self.coef.shape[-1] - 1),
                           dtype=complex)

        # Iterate over polynomial degrees.
        is_done = numpy.broadcast_to(False, self.coef.shape[:-1])
        num_done = 0
        num_poly = is_done.size

        # Degrees above 0
        for d in range(self.coef.shape[-1] - 1, 0, -1):
            # Find polynomials of given degree.
            flg = self.coef[..., d] != 0 & ~is_done

            # Build companion matrices
            sub_coef = self.coef[flg]
            mat = numpy.empty((*sub_coef.shape[:-1], d, d))
            mat[..., -1] = -sub_coef[..., :d] / sub_coef[..., d, numpy.newaxis]
            mat[..., :-1] = 0.0

            idx = numpy.arange(-1, d - 1)[*(numpy.newaxis
                                            for _ in mat.shape[:-2]),
                                          :]
            idx[..., 0] = 0
            val = numpy.ones((d,))[*(numpy.newaxis for _ in mat.shape[:-2]),
                                   :]
            val[..., 0] = 0.0
            numpy.put_along_axis(mat, idx[..., numpy.newaxis],
                                 val[..., numpy.newaxis], axis=-1)

            # Solve for the eigenvalues of the companion matrix.
            root[flg, :d] = numpy.linalg.eigvals(mat)
            root[flg, d:] = fill

            # Update done count
            is_done = is_done | flg
            num_done += numpy.sum(flg)
            if num_done == num_poly:
                break

        # Fill out roots of zero-degree polynomials.
        root[~is_done, :] = fill

        # Convert roots back to original space.
        root = (
            (
                (root - self.window[0])
                / self._scale[:, numpy.newaxis]
            )
            + self.ts[:-1, numpy.newaxis]
        )

        return root

    def convert(self, *, window: Optional[tuple[float, float]] = None,
                time_grid: Optional[ArrayLike] = None
                ) -> 'PolynomialTrajectory':
        '''
        Convert to a different parameter window or time grid.

        :param window: New parameter window. The old window is used by
                       default.
        :type window: tuple of 2 floats, optional
        :param time_grid: New time grid. Defaults to the old time grid.
        :type time_grid: 1-D array-like, optional

        ..warning::
            The output of this method is undefined if `time_grid` is
            not a refinement of the intersection of the trajectory's
            prior time grid with the timespan of the new time grid,
            i.e., if `time_grid` skips over one of the times in the old
            time grid.
        '''
        # Short circuit if no change is necessary.
        if (
                (window is None or window == self.window)
                and (time_grid is None
                     or numpy.array_equiv(time_grid, self.ts))
        ):
            return self

        # Substitute default values for arguments.
        if window is None:
            window = self.window
        if time_grid is None:
            time_grid = self.ts

        # Convert time grid into a NumPy array.
        time_grid = numpy.asarray(time_grid)

        # Find relevant intervals.
        ind = numpy.searchsorted(self.ts[1:], time_grid[:-1], side='right') - 1
        ind = numpy.fmax(ind, 0)

        # Set up the coefficient array and find parameters for start and end.
        coef = self.coef[ind, ...]
        n_coef = coef.shape[-1]
        x0 = (time_grid[:-1] - self.ts[ind]) * self._scale[ind]
        x1 = (time_grid[1:] - self.ts[ind]) * self._scale[ind]

        # FIXME: Static type checker becomes confused about data types here.
        x0 = cast(numpy.ndarray, x0)
        x1 = cast(numpy.ndarray, x1)

        # Calculate translation from new to old parameter window.
        y0, y1 = window
        scale = (x1 - x0) / (y1 - y0)
        shift = x0 - scale * y0

        scale_mat = numpy.cumprod(
            numpy.flipud(numpy.tri(n_coef - 1))[
                *(numpy.newaxis for _ in coef.shape[:-2]), ...
            ]
            * scale[:, *(numpy.newaxis for _ in coef.shape[1:])],
            axis=-1
        )

        shift_mat = numpy.flip(
            numpy.cumprod(
                numpy.flip(
                    numpy.flipud(numpy.tri(n_coef - 1))[
                        *(numpy.newaxis for _ in coef.shape[:-2]), ...
                    ]
                    * shift[:, *(numpy.newaxis for _ in coef.shape[1:])],
                    axis=-1
                ),
                axis=-1
            ),
            axis=-1
        )

        # Build change-of-parameter matrix.
        mat = scipy.special.comb(
            numpy.arange(n_coef - 1, -1, -1)[
                *repeat(numpy.newaxis, n_coef - 2), :, numpy.newaxis
            ],
            numpy.arange(n_coef)[*repeat(numpy.newaxis, n_coef - 1), :]
        )
        mat[..., :-1, 1:] *= scale_mat
        mat[..., :-1, :-1] *= shift_mat

        # Return result.
        return PolynomialTrajectory(time_grid, coef @ mat, window=window)

    def __mul__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            return PolynomialTrajectory(self.ts, self.coef * arg,
                                        window=self.window,
                                        mapscale=self._scale)
        return NotImplemented

    def __rmul__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            return PolynomialTrajectory(self.ts, self.coef * arg,
                                        window=self.window,
                                        mapscale=self._scale)
        return NotImplemented

    def __truediv__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            return PolynomialTrajectory(self.ts, self.coef / arg,
                                        window=self.window,
                                        mapscale=self._scale)
        return NotImplemented

    def __neg__(self) -> 'PolynomialTrajectory':
        return PolynomialTrajectory(self.ts, -self.coef,
                                    window=self.window,
                                    mapscale=self._scale)

    def __add__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            coef = numpy.array(self.coef)
            coef[..., 0] += arg
            return PolynomialTrajectory(self.ts, coef,
                                        window=self.window,
                                        mapscale=self._scale)
        if isinstance(arg, PolynomialTrajectory):
            if arg.window != self.window or not numpy.array_equiv(arg.ts,
                                                                  self.ts):
                joint_time = sortednp.merge(arg.ts, self.ts,
                                            duplicates=sortednp.DROP)
                self = self.convert(time_grid=joint_time)
                arg = arg.convert(window=self.window, time_grid=joint_time)
            return PolynomialTrajectory(self.ts, self.coef + arg.coef,
                                        window=self.window,
                                        mapscale=self._scale)
        return NotImplemented

    def __radd__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int, PolynomialTrajectory)):
            return self.__add__(arg)
        return NotImplemented

    def __sub__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            coef = numpy.array(self.coef)
            coef[..., 0] -= arg
            return PolynomialTrajectory(self.ts, coef,
                                        window=self.window,
                                        mapscale=self._scale)
        if isinstance(arg, PolynomialTrajectory):
            if arg.window != self.window or not numpy.array_equiv(arg.ts,
                                                                  self.ts):
                joint_time = sortednp.merge(arg.ts, self.ts,
                                            duplicates=sortednp.DROP)
                self = self.convert(time_grid=joint_time)
                arg = arg.convert(window=self.window, time_grid=joint_time)
            return PolynomialTrajectory(self.ts, self.coef - arg.coef,
                                        window=self.window,
                                        mapscale=self._scale)
        return NotImplemented

    def __rsub__(self, arg) -> 'PolynomialTrajectory | NotImplementedType':
        if isinstance(arg, (float, int)):
            coef = -self.coef
            coef[..., 0] += arg
            return PolynomialTrajectory(self.ts, coef,
                                        window=self.window,
                                        mapscale=self._scale)
        if isinstance(arg, PolynomialTrajectory):
            return arg.__sub__(self)
        return NotImplemented


def midpoint_time_grid(time_grid: ArrayLike) -> numpy.ndarray:
    '''
    Refine a time grid by adding the midpoint of each time interval.
    '''
    time_grid = numpy.asarray(time_grid)
    midpoint_grid = numpy.empty((2 * time_grid.size - 1,))
    midpoint_grid[::2] = time_grid
    midpoint_grid[1::2] = (time_grid[1:] + time_grid[:-1]) / 2
    return midpoint_grid


def merge_time_grids(*grids: ArrayLike) -> numpy.ndarray:
    '''
    Join multiple time grids into one that is a refinement of all inputs.
    '''
    grid_arrays = [numpy.asarray(grid) for grid in grids]
    merge_grid = sortednp.kway_merge(*grid_arrays)
    repeat_idx = numpy.nonzero(merge_grid[1:] == merge_grid[:-1])
    return numpy.delete(merge_grid, repeat_idx)


def polyfit_quartic(time: ArrayLike, value: ArrayLike, deriv: ArrayLike
                    ) -> PolynomialTrajectory:
    '''
    Fit a spline of quartic polynomials to calculated values and derivatives.

    This fits a sequence of quartic polynomials to precalculated values and
    derivatives of a continuously differentiable function on a given time
    grid. Values are calculated at the endpoints and the midpoint of each
    interval, derivatives are evaluated only at the endpoints.

    The function can cope with two types of time grid inputs. If only the
    endpoint times of the intervals are given, then it will compute the
    midpoint times on its own.

    :param time: Time grid. Can be of shape `(n,)` or `(2n - 1,)` where is the
                 number of interval endpoints, depending on whether or not
                 midpoints are pre-calculated. Must be sorted in ascending
                 order.
    :type time: array-like, shape `(n,)` or `(2n - 1,)`
    :param value: End- and midpoint values. Columns must be ordered by
                  ascending time value. Must be of shape `(m, 2n - 1)` or,
                  possibly, `(2n - 1,)` if `m == 1`.
    :type value: array-like, shape `(m, 2n - 1)`
    :param deriv: Endpoint derivatives. Columns must be ordered by ascending
                  time value. Must be of shape `(m, n)` or, possibly, `(n,)`
                  if `m == 1`.
    :type deriv: array-like, shape `(m, n)`

    :return: An object of type :class:`PolynomialTrajectory`.
    '''
    # Convert inputs to numpy.ndarray.
    time = numpy.asarray(time)
    value = numpy.asarray(value)
    deriv = numpy.asarray(deriv)

    # Reshape if m == 1.
    if deriv.ndim == 1:
        deriv = deriv[numpy.newaxis, :]
    if value.ndim == 1:
        value = value[numpy.newaxis, :]
    if time.ndim != 1:
        raise ValueError('`time` must be a 1-D array')

    # Ensure that all arrays are properly shaped.
    n_comp, n_time = deriv.shape
    if value.shape != (n_comp, 2 * n_time - 1):
        raise ValueError(f'`value.shape` is {value.shape}, which is '
                         'incompatible with `deriv.shape`, which is '
                         f'{deriv.shape}')
    if time.shape == (n_comp, n_time):
        midpoint_time = numpy.empty((2 * n_time - 1,))
        midpoint_time[::2] = time
        midpoint_time[1::2] = (time[1:] + time[:-1]) / 2
        time = midpoint_time
    if time.shape != (2 * n_time - 1,):
        raise ValueError(f'`time.shape` must be either ({n_time},) or '
                         f'({2 * n_time - 1},), but is {time.shape}.')

    # Calculate the half step of each interval.
    half_step = ((time[2::2] - time[:-2:2]) / 2)[:, numpy.newaxis]

    # Fit quartic polynomials for each interval.
    poly_coeff = numpy.empty((n_time - 1, n_comp, 5))
    poly_coeff[:, :, 0] = value[:, 1::2].T
    poly_coeff[:, :, 1] = (value[:, 2::2] + value[:, :-2:2]
                           - 2 * value[:, 1::2]).T / 2
    poly_coeff[:, :, 2] = (value[:, 2::2] - value[:, :-2:2]).T / 2
    poly_coeff[:, :, 3] = (deriv[:, 1:] + deriv[:, :-1]).T * half_step / 2
    poly_coeff[:, :, 4] = (deriv[:, 1:] - deriv[:, :-1]).T * half_step / 4

    mat = numpy.array([[[
        [1.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  1.5, -0.5,  0.0],
        [0.0,  2.0,  0.0,  0.0, -1.0],
        [0.0,  0.0, -0.5,  0.5,  0.0],
        [0.0, -1.0,  0.0,  0.0,  1.0],
    ]]])
    poly_coeff = numpy.squeeze(mat @ poly_coeff[..., numpy.newaxis], -1)

    # Generate polynomial objects for each interval.
    return PolynomialTrajectory(time[::2], poly_coeff)
