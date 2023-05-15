'''
Implementation of a similarity class.
'''

import copy
import functools
from types import NotImplementedType
from typing import Callable, Optional, Self

import numpy
from numpy.typing import ArrayLike
import sortednp

from pycoimset.typing import SignedMeasure, SimilarityClass, SimilaritySpace


__all__ = [
    'IntervalSimilarityClass',
    'IntervalSimilaritySpace',
    'PolynomialSignedMeasure',
]


def filter_switch_time_duplicates(times: ArrayLike):
    '''Filter duplicate entries out of a sorted switching time array.'''
    # Convert input to NumPy array.
    times = numpy.asarray(times)

    # Indicates whether an element is equal to its successor.
    eq_flg = numpy.concatenate((times[:-1] == times[1:], [False]))

    # Indicates whether eq_flg has changed from the predecessor.
    chg_flg = numpy.concatenate(([eq_flg[0]], eq_flg[:-1] != eq_flg[1:]))

    # Find indices where runs start or end.
    chg_idx = numpy.flatnonzero(chg_flg).reshape((-1, 2))

    # Find chunks of the times vector to keep.
    chunks = []
    chunk_idx = 0
    for start, end in chg_idx:
        end = start + 2 * ((end - start + 1) // 2)
        if start < end:
            chunks.append(times[chunk_idx:start])
            chunk_idx = end

    if chunk_idx < len(times):
        chunks.append(times[chunk_idx:])

    return numpy.concatenate(chunks)


def filter_single_pair_duplicates(times: ArrayLike):
    '''Filter duplicates out of sorted switch times with only single pairs.'''
    # Convert input to numpy array.
    times = numpy.asarray(times)

    # Find equal elements.
    eq_idx = numpy.flatnonzero(times[1:] == times[:-1])

    # Build deletion index.
    del_idx = numpy.empty(len(eq_idx) * 2, dtype=int)
    del_idx[::2] = eq_idx
    del_idx[1::2] = eq_idx + 1

    # Return result.
    return numpy.delete(times, del_idx)


def coerce_polynomial_spline(ps: ArrayLike, ts: ArrayLike, ind: ArrayLike
                             ) -> numpy.ndarray:
    '''Coerce a polynomial spline into a given time grid.'''
    # Import arrays.
    ps = numpy.asarray(ps)
    ts = numpy.asarray(ts)
    ind = numpy.asarray(ind)

    # Coerce spline.
    size = ind[1:] - ind[:-1]
    pout = numpy.empty(len(ts) - 1, dtype=object)
    pout[ind[flag]] = ps[flag := size == 1]
    for idx in numpy.flatnonzero(size > 1):
        tss = ts[ind[idx]:ind[idx+1]]
        pin = ps[idx]
        pss = [
            pin.convert(domain=dom)             # type: ignore
            for dom in zip(tss[:-1], tss[1:])
        ]
        pout[ind[idx]:ind[idx+1]] = pss

    return pout


def join_polynomial_splines(ta: ArrayLike, pa: ArrayLike, tb: ArrayLike,
                            pb: ArrayLike) -> tuple[numpy.ndarray,
                                                    numpy.ndarray,
                                                    numpy.ndarray]:
    '''Adapt time grids of two polynomial splines.'''
    # Import polynomial arrays.
    pa = numpy.asarray(pa)
    pb = numpy.asarray(pb)

    # Join time grids.
    t, (ia, ib) = sortednp.merge(ta, tb, indices=True,
                                 duplicates=sortednp.DROP)

    # Coerce splines.
    pa_out = coerce_polynomial_spline(pa, t, ia)
    pb_out = coerce_polynomial_spline(pb, t, ib)

    return t, pa_out, pb_out


class IntervalSimilaritySpace(SimilaritySpace['IntervalSimilaritySpace']):
    time_range: tuple[float, float]

    def __init__(self, time_range: tuple[float, float]):
        start, end = time_range
        if start >= end:
            raise ValueError('time range cannot be empty')
        self.time_range = time_range

    def __repr__(self) -> str:
        '''Generate string representation.'''
        return f'IntervalSimilaritySpace({repr(self.time_range)})'

    @property
    def measure(self) -> float:
        '''Return measure of interval.'''
        start, end = self.time_range
        return end - start

    @property
    def empty_class(self) -> 'IntervalSimilarityClass':
        '''Return empty similarity class.'''
        return IntervalSimilarityClass(self, [])

    @property
    def universal_class(self) -> 'IntervalSimilarityClass':
        '''Return universal similarity class.'''
        return IntervalSimilarityClass(self, self.time_range)


class IntervalSimilarityClass(SimilarityClass[IntervalSimilaritySpace]):
    #: Underlying similarity space.
    _space: IntervalSimilaritySpace

    #: Array of switching times.
    switch_times: numpy.ndarray

    def __init__(self, space: IntervalSimilaritySpace,
                 switch_times: ArrayLike,
                 *,
                 sort: bool = False, filter: bool = False):
        '''
        Create new similarity class.

        This method can be used to sort and filter switching times. Note that,
        because switching times are matched in start-end pairs, they must
        always be removed in pairs. Therefore, filtering cannot be achieved
        by simply calling `numpy.unique`.

        :param space: Reference to the underlying similarity space.
        :type space: :class:`IntervalSimilaritySpace`
        :param switch_times: Switching times that delineate the similarity
                             class.
        :type switch_times: array-like
        :param sort: Indicates that `switch_times` is not pre-sorted. Defaults
                     to `False`.
        :type sort: bool
        :param filter: Indicates that the entries of `switch_times` are not
                       unique and therefore may require filtering. Defaults to
                       `False`.
        :type filter: bool
        '''
        # Set underlying measure space.
        self._space = space

        # Obtain time range.
        start, end = space.time_range

        # Convert switch times to array and sort.
        switch_times = numpy.asarray(switch_times, dtype=float)
        if sort:
            switch_times = numpy.sort(switch_times)

        # Ensure even number of switch times by inserting end of universal
        # time interval if necessary.
        if len(switch_times) % 2 == 1:
            if switch_times[-1] == end:
                switch_times = switch_times[:-1]
            else:
                switch_times = numpy.concatenate((switch_times, [end]))

        # Filter switching times outside the universal time range.
        if filter:
            # Filter entries outside the universal time range.
            start_idx, end_idx = numpy.searchsorted(switch_times,
                                                    [start, end],
                                                    side='right')
            arr: list[ArrayLike] = [switch_times[start_idx:end_idx]]
            if start_idx % 2 == 1:
                arr.insert(0, [start])
            if end_idx % 2 == 1:
                arr.append([end])
            switch_times = numpy.concatenate(arr)

            # Remove duplicate entries.
            switch_times = filter_switch_time_duplicates(switch_times)

        self.switch_times = switch_times

    def __repr__(self) -> str:
        '''Generate string representation.'''
        return (f'IntervalSimilarityClass({repr(self._space)}, '
                f'{repr(self.switch_times)})')

    @property
    def space(self) -> IntervalSimilaritySpace:
        '''Underlying similarity space.'''
        return self._space

    @property
    def measure(self) -> float:
        '''Measure of the class.'''
        return numpy.sum(self.switch_times[1::2] - self.switch_times[::2])

    def __copy__(self) -> 'IntervalSimilarityClass':
        return IntervalSimilarityClass(self._space, self.switch_times)

    def __deepcopy__(self) -> 'IntervalSimilarityClass':
        return IntervalSimilarityClass(
            self._space, copy.deepcopy(self.switch_times)
        )

    def subset(self, meas_low: float, meas_high: float,
               hint: Optional[SignedMeasure] = None
               ) -> 'IntervalSimilarityClass':
        cum_meas = numpy.cumsum(self.switch_times[1:] - self.switch_times[:-1])
        idx = numpy.searchsorted(cum_meas, meas_high, side='right')
        times: list[ArrayLike] = [self.switch_times[:2 * idx - 1]]
        times.append([meas_high - cum_meas[idx - 1]
                      + self.switch_times[2 * idx - 2]])
        switch_times = numpy.concatenate(times)
        return IntervalSimilarityClass(self._space, switch_times)

    def __invert__(self) -> Self:
        '''Return complement of this class.'''
        # Complement is formed by adding the start and end times of the
        # universal time interval to the switching time vector.
        start, end = self._space.time_range

        # Set up new switching time vector.
        switch_times = self.switch_times
        if len(switch_times) > 0 and switch_times[0] == start:
            switch_times = switch_times[1:]
            front = []
        else:
            front = [start]
        if len(switch_times) > 0 and switch_times[-1] == end:
            switch_times = switch_times[:-1]
            back = []
        else:
            back = [end]
        switch_times = numpy.concatenate((front, switch_times, back))

        return IntervalSimilarityClass(self._space, switch_times)

    def __or__(self, other: SimilarityClass) -> Self | NotImplementedType:
        '''Return union with another similarity class.'''
        if not isinstance(other, IntervalSimilarityClass):
            return NotImplemented
        if other._space is not self._space:
            raise ValueError('`other` is not within the same similarity space')

        # Get switching times such that `switch_a` is the smaller of the two
        # intervals.
        switch_a = self.switch_times
        switch_b = other.switch_times

        if len(switch_a) == 0:
            return other
        if len(switch_b) == 0:
            return self

        if len(switch_a) > len(switch_b):
            switch_a, switch_b = switch_b, switch_a

        # Find insertion points for `switch_a` in `switch_b`
        ins_idx = numpy.searchsorted(switch_b, switch_a, side='left'
                                     ).reshape((-1, 2))
        even_flg = ins_idx % 2 == 0
        keep_idx = numpy.concatenate(([0], ins_idx.flatten(), [len(switch_b)])
                                     ).reshape((-1, 2))

        # Remove all covered switch times of the `b` class.
        chunks_b = [switch_b[start:end] for start, end in keep_idx]
        chunks_a = [[], *(row[flg] for row, flg
                          in zip(switch_a.reshape((-1, 2)), even_flg))]

        # Merge the chunk lists.
        chunks = [chunk for chunk_pair in zip(chunks_a, chunks_b)
                  for chunk in chunk_pair if len(chunk) > 0]
        switch_times = numpy.concatenate(chunks)

        # Remove duplicates (can only be single pair duplicates)
        switch_times = filter_single_pair_duplicates(switch_times)

        return IntervalSimilarityClass(self._space, switch_times)

    def __and__(self, other: SimilarityClass) -> Self | NotImplementedType:
        if not isinstance(other, IntervalSimilarityClass):
            return NotImplemented
        if other.space is not self.space:
            raise ValueError('`other` is not within the same similarity space')

        return ~(~self | ~other)

    def __sub__(self, other: SimilarityClass) -> Self | NotImplementedType:
        if not isinstance(other, IntervalSimilarityClass):
            return NotImplemented
        if other.space is not self.space:
            raise ValueError('`other` is not within the same similarity space')

        return self & ~other

    def __rsub__(self, other: SimilarityClass) -> Self | NotImplementedType:
        if not isinstance(other, IntervalSimilarityClass):
            return NotImplemented
        return other.__sub__(self)

    def __xor__(self, other: SimilarityClass) -> Self | NotImplementedType:
        if not isinstance(other, IntervalSimilarityClass):
            return NotImplemented
        if other.space is not self.space:
            raise ValueError('`other` is not within the same similarity space')

        switch_times = sortednp.merge(
            self.switch_times, other.switch_times
        )

        return IntervalSimilarityClass(self._space, switch_times, filter=True)


class PolynomialSignedMeasure(SignedMeasure[IntervalSimilaritySpace]):
    '''
    Signed measure encoded as an array of polynomials.
    '''

    #: Underlying measure space.
    _space: IntervalSimilaritySpace

    #: Array of interpolation polynomials.
    polynomials: numpy.ndarray

    #: NumPy ufunc to get lower domain bound of polynomials (returns dtype
    #: 'object').
    __polydomlb = numpy.frompyfunc(lambda p: p.domain[0], nin=1, nout=1)

    #: NumPy ufunc to get derivative of polynomials.
    __polyderiv = numpy.frompyfunc(lambda p: p.deriv(), nin=1, nout=1)

    #: NumPy ufunc to get integral of polynomials.
    __polyinteg = numpy.frompyfunc(lambda p: p.integ(), nin=1, nout=1)

    def __init__(self, space: IntervalSimilaritySpace,
                 polynomials: numpy.ndarray | list[object]):
        self._space = space
        self.polynomials = numpy.asarray(polynomials, dtype=object)

    def __repr__(self) -> str:
        '''Generate string representation.'''
        return (f'PolynomialSignedMeasure({repr(self._space)}, '
                f'{repr(self.polynomials)})')

    @property
    def space(self) -> IntervalSimilaritySpace:
        '''Underlying similarity space.'''
        return self._space

    @functools.cached_property
    def time_grid(self) -> numpy.ndarray:
        '''Starting times for the polynomial domains.'''
        if len(self.polynomials) == 0:
            return numpy.empty((0,))
        return PolynomialSignedMeasure.__polydomlb(self.polynomials)\
                                      .astype(float)

    @functools.cached_property
    def poly_integ(self) -> numpy.ndarray:
        '''Array of integral polynomials.'''
        return PolynomialSignedMeasure.__polyinteg(self.polynomials)

    @functools.cached_property
    def poly_deriv(self) -> numpy.ndarray:
        '''Array of integral polynomials.'''
        return PolynomialSignedMeasure.__polyderiv(self.polynomials)

    def __call__(self, simcls: SimilarityClass) -> float:
        '''Return measure of a similarity class.'''
        if not isinstance(simcls, IntervalSimilarityClass):
            raise TypeError('`simcls` must be of type '
                            '`IntervalSimilarityClass`')
        if simcls.space is not self._space:
            raise ValueError('Similarity space does not match.')

        # Retrieve integral polynomials and time grid.
        start_times = self.time_grid
        poly = self.poly_integ
        switch = simcls.switch_times

        # Short-circuit if there is nothing to do. Otherwise, obtain end of
        # last polynomial's domain.
        if len(poly) == 0:
            return 0.0
        end_time = poly[-1].domain[1]

        # Locate switching times within starting time array.
        pos_start = numpy.searchsorted(start_times, switch[0::2], side='right'
                                       ) - 1
        pos_end = numpy.searchsorted(start_times, switch[1::2], side='left')

        row_nonzero = (pos_end > 0) & (switch[0::2] < end_time)
        row_single = row_nonzero & (pos_start + 1 == pos_end)
        row_multi = row_nonzero & ~row_single

        idx_single = numpy.flatnonzero(row_single)
        idx_multi = numpy.flatnonzero(row_multi)

        # Accumulate measure.
        meas_single = sum((
            (p := poly[pos_start[ridx]])(switch[2 * ridx + 1])  # type: ignore
            - p(switch[2 * ridx])                               # type: ignore
            for ridx in idx_single
        ))
        meas_multi_inner = sum((
            (p := poly[idx])(p.domain[1])                       # type: ignore
            - p(p.domain[0])                                    # type: ignore
            for ridx in idx_multi
            for idx in range(pos_start[ridx] + 1, pos_end[ridx] - 1)
        ))
        meas_multi_start = sum((
            (p := poly[idx])(p.domain[1])                       # type: ignore
            - p(switch[2 * ridx])                               # type: ignore
            for ridx in idx_multi
            if (idx := pos_start[ridx]) >= 0
        ))
        meas_multi_end = sum((
            (p := poly[pos_end[ridx] - 1])(                     # type: ignore
                switch[2 * ridx + 1]
            )
            - p(p.domain[0])                                    # type: ignore
            for ridx in idx_multi
        ))

        return (meas_single + meas_multi_inner + meas_multi_start
                + meas_multi_end)

    def __levelset(self, cmp: Callable[[ArrayLike, ArrayLike], ArrayLike],
                   level: float) -> IntervalSimilarityClass:
        '''Obtain sublevel set for shifted polynomials.'''
        # Obtain shifted polynomials and derivative polynomials.
        poly = self.polynomials

        # Find roots of each polynomial.
        roots = numpy.stack([
            numpy.concatenate((p.domain, (p - level).roots())) for p in poly
        ])
        dom_start, dom_end = roots[:, 0], roots[:, 1]
        roots = numpy.real_if_close(roots)

        # Sort roots in ascending order with complex roots at the end.
        ind = numpy.lexsort(
            (roots.real, abs(roots.imag)), axis=-1
        )
        roots = numpy.take_along_axis(roots, ind, axis=-1)

        # For each interval, find the range of real roots inside the domain.
        # NOTE: At this point, we discard the imaginary part.
        is_real, roots = roots.imag == 0.0, numpy.array(roots.real)
        is_after_start = roots >= dom_start.reshape((-1, 1))
        is_before_end = roots <= dom_end.reshape((-1, 1))
        within_domain = is_real & is_after_start & is_before_end
        mid_within_domain = within_domain[:, :-1] & within_domain[:, 1:]

        # Calculate midpoints (real axis only) and eval shifted polynomials.
        mid = (roots[:, 1:] + roots[:, :-1]) / 2
        mid_val = numpy.stack([p(x) for p, x in zip(poly, mid)])

        # Apply comparator to all midpoint values.
        mid_match = numpy.asarray(cmp(mid_val, level))

        # Figure out which interval break points should be included in
        # the list of switch times.
        mid_match = numpy.concatenate((
            [False],
            mid_match.flatten()[mid_within_domain.flatten()],
            [False]
        ))
        switch_flag = mid_match[1:] != mid_match[:-1]

        # Generate a list of switching times to apply `switch_flag` to.
        # This requires removing the first time point of each
        # polynomial starting with the second row because it duplicates
        # the end point of the previous polynomial's domain.
        start_idx = numpy.sum(is_real & ~is_after_start, axis=-1,
                              keepdims=True)
        numpy.put_along_axis(within_domain[1:, :], start_idx[1:, :], False,
                             axis=1)
        switch_times = roots.flatten()[within_domain.flatten()][switch_flag]

        return IntervalSimilarityClass(self._space, switch_times, filter=True)

    def __lt__(self, level: float) -> IntervalSimilarityClass:
        return self.__levelset(numpy.less, level)

    def __le__(self, level: float) -> IntervalSimilarityClass:
        return self.__levelset(numpy.less_equal, level)

    def __gt__(self, level: float) -> IntervalSimilarityClass:
        return self.__levelset(numpy.greater, level)

    def __ge__(self, level: float) -> IntervalSimilarityClass:
        return self.__levelset(numpy.greater_equal, level)

    def __mul__(self, factor: float) -> Self:
        return PolynomialSignedMeasure(self._space, self.polynomials * factor)

    def __rmul__(self, factor: float) -> Self:
        return PolynomialSignedMeasure(self._space, self.polynomials * factor)

    def __add__(self, other: SignedMeasure) -> Self | NotImplementedType:
        if not isinstance(other, PolynomialSignedMeasure):
            return NotImplemented
        if other._space is not self._space:
            raise ValueError('Similarity space mismatch')

        _, pa, pb = join_polynomial_splines(
            self.time_grid, self.polynomials,
            other.time_grid, other.polynomials
        )

        return PolynomialSignedMeasure(self._space, pa + pb)

    def __radd__(self, other: SignedMeasure) -> Self | NotImplementedType:
        if not isinstance(other, PolynomialSignedMeasure):
            return NotImplemented
        return self.__add__(other)

    def __sub__(self, other: SignedMeasure) -> Self | NotImplementedType:
        if not isinstance(other, PolynomialSignedMeasure):
            return NotImplemented
        if other._space is not self._space:
            raise ValueError('Similarity space mismatch')

        _, pa, pb = join_polynomial_splines(
            self.time_grid, self.polynomials,
            other.time_grid, other.polynomials
        )

        return PolynomialSignedMeasure(self._space, pa - pb)

    def __rsub__(self, other: SignedMeasure) -> Self | NotImplementedType:
        if not isinstance(other, PolynomialSignedMeasure):
            return NotImplemented
        return other.__sub__(self)
