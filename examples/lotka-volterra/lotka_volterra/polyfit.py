'''
Helpers for polynomial fitting of 'fake' ODE trajectories.
'''

import copy

import numpy
from numpy.typing import ArrayLike, NDArray
import scipy.integrate
import sortednp


__all__ = [
    'PolynomialDenseOutput',
    'merge_time_grids',
    'midpoint_time_grid',
    'polyfit_quartic',
]


class PolynomialDenseOutput(scipy.integrate.DenseOutput):
    '''
    Dense output representation using NumPy polynomials for interpolation.
    '''

    #: Array of component polynomials.
    polynomials: NDArray

    def __init__(self, poly: ArrayLike):
        '''
        Create dense output from an array-like of polynomials.

        All polynomials should have a shared domain. The domain of the
        `DenseOutput` object is the intersection of all polynomial domains.

        :param poly: array-like of polynomials.
        :type poly: array-like of polynomials
        '''
        poly = numpy.asarray(poly).flatten()
        t_start = max((p.domain[0] for p in poly))
        t_end = min((p.domain[1] for p in poly))
        super().__init__(t_start, t_end)

        self.polynomials = copy.copy(poly)

    def __call__(self, time: ArrayLike) -> NDArray:
        '''Evaluate component polynomials at times.'''
        time = numpy.asarray(time)
        return numpy.stack([p(time) for p in self.polynomials])

    def deriv(self, time: ArrayLike) -> NDArray:
        '''Evaluate derivative of polynomials at times.'''
        time = numpy.asarray(time)
        return numpy.stack([p.deriv()(time) for p in self.polynomials])


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
                    ) -> scipy.integrate.OdeSolution:
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

    :return: An `OdeSolution` object of :class:`PolynomialDenseOutput`
             objects.
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
    poly_arr = numpy.array([
        [
            numpy.polynomial.Polynomial(coef, window=(-1, 1), domain=(t0, t1))
            for coef in poly_slice
        ]
        for poly_slice, t0, t1 in zip(poly_coeff, time[:-2:2], time[2::2])
    ])
    return scipy.integrate.OdeSolution(
        time[::2], [PolynomialDenseOutput(poly) for poly in poly_arr]
    )
