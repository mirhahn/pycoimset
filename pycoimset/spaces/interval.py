# PyCoimset - Python library for optimization with set-valued variables.
# Copyright 2025 Mirko Hahn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Borel measurable subsets of a finite interval with switching time vectors as
set encoding.
"""

from collections.abc import Callable
from types import NotImplementedType
from typing import Self, TypeVar

import numpy
from numpy.typing import ArrayLike, NDArray

from ..typing.space import SignedMeasure, SimilarityClass, SimilaritySpace
from .generic import EmptyClass, UniversalClass


Spc = TypeVar('Spc', bound='IntervalBorelSpace')


class IntervalBorelSpace(SimilaritySpace):
    """
    Borel sigma-algebra on a compact interval of real numbers.
    """
    def __init__(self, a: float, b: float, set_func: Callable[
                    [float, float, SignedMeasure[Self] | None],
                    SimilarityClass[Self]
                 ] | None = None):
        super().__init__()
        self._lb = a
        self._ub = b
        self._set = set_func

    @property
    def bounds(self) -> tuple[float, float]:
        return self._lb, self._ub

    @property
    def measure(self) -> float:
        return max(0.0, self._ub - self._lb)

    @property
    def empty_class(self) -> EmptyClass[Self]:
        return EmptyClass[Self](self)

    @property
    def universal_class(self) -> UniversalClass[Self]:
        return UniversalClass[Self](self, self.pick_set)

    def pick_set(self, meas_low: float, meas_high: float,
                 hint: SignedMeasure[Self] | None = None
                 ) -> SimilarityClass[Self]:
        """
        Pick an arbitrary similarity class in the given measure range.

        Args:
            meas_low: Lower bound on the measure of the class.
            meas_high: Upper bound on the measure of the class.
            hint: Signed measure within the same space that the function
                should attempt to minimize. Currently ignored.

        Returns:
            A similarity class within this space that is within the given
            measure range.

        Raises:
            ValueError: There exists no similarity class within the measure
                range.
        """
        # Try to use user-specified subset picker.
        if (set_func := self._set) is not None:
            return set_func(meas_low, meas_high, hint)

        # Check whether request is satisfiable.
        spc_meas = self.measure
        if (
            meas_low > min(spc_meas, meas_high)
            or meas_high < max(0.0, meas_low)
        ):
            raise ValueError("no similarity class in measure range")

        # Short-circuit if the empty or universal class are an option.
        if meas_low <= 0.0:
            return self.empty_class
        if meas_high >= spc_meas:
            return self.universal_class

        # Pick from the end of the interval.
        return SwitchTimeClass[Self](
            self, [max(self._ub - meas_high, self._lb), self._ub]
        )


class SwitchTimeClass(SimilarityClass[Spc]):
    """
    Similarity class encoded as a sorted vector of distinct switching times.
    """

    @staticmethod
    def _deduplicate(time_vec: NDArray[numpy.float64]
                     ) -> NDArray[numpy.float64]:
        """
        Eliminate pairs of duplicate times from sorted time vector.

        Args:
            time_vec: Sorted flat array of switching times. Must have even
                number of entries.

        Returns:
            An array of switching times created by removing pairs of equal
            switching times from `time_vec` until all remaining times are
            unique.
        """
        while numpy.any(mask := time_vec[1:] == time_vec[:-1]):
            mask = numpy.append(mask, [False])
            if numpy.any(mask[:-2:2]):
                mask[1::2] = mask[:-2:2]
            else:
                mask[2::2] = mask[1::2]
            time_vec = numpy.delete(time_vec, numpy.flatnonzero(mask))
        return time_vec

    def __init__(self, space: Spc, times: ArrayLike):
        super().__init__()
        self._space = space

        # Prepare switching times vector.
        lb, ub = space.bounds
        if ub < lb:
            times = numpy.empty(0, dtype=numpy.float64)
        else:
            times = numpy.sort(
                numpy.asarray(times, dtype=numpy.float64).flatten()
            )
            times = numpy.clip(times, lb, ub)
            if len(times) % 2 != 0:
                times = numpy.append(times, [ub])
        self._times = self._deduplicate(times)

    @property
    def space(self) -> Spc:
        return self._space

    @property
    def measure(self) -> float:
        t = self._times
        return float(numpy.sum(t[1::2] - t[::2]))

    def subset(self, meas_low: float, meas_high: float,
               hint: SignedMeasure[Spc] | None = None
               ) -> Self:
        # Calculate measures of individual intervals.
        t = self._times
        dt = t[1::2] - t[::2]

        # Sort intervals by size (ascending).
        idx_sort = numpy.argsort(dt)

        # Calculate cumulative measure and find break point.
        dt_cum = numpy.cumsum(dt[idx_sort])
        idx_brk = numpy.searchsorted(dt_cum, meas_high, side='right')

        # At this point, the first `idx_brk` intervals in the sorted list
        # should be included. The interval with index `idx_brk` should be
        # included partially.
        idx_full = idx_sort[:idx_brk]
        idx_partial = None if idx_brk >= len(idx_sort) else idx_sort[idx_brk]

        # Build filter mask for switching time vector.
        mask = numpy.zeros_like(t, dtype=numpy.bool_)
        mask[2 * idx_full] = True
        mask[2 * idx_full + 1] = True

        # Calculate remaining measure.
        base_measure = 0.0 if idx_brk == 0 else dt_cum[idx_brk - 1]

        # Fill with partial interval if necessary.
        if base_measure < meas_low and idx_partial is not None:
            t = numpy.copy(t)
            mask[2 * idx_partial] = True
            mask[2 * idx_partial + 1] = True
            t[2 * idx_partial] = (
                t[2 * idx_partial + 1] - (meas_high - base_measure)
            )

        # Create a reduced switching time vector.
        t = t[mask]

        # Return result.
        return type(self)(self._space, t)

    def __invert__(self) -> Self:
        # Add switching times at start and end of the switching time vector.
        lb, ub = self._space.bounds
        return type(self)(
            self._space, numpy.concatenate((lb, self._times, ub))
        )

    def __or__(self, b: SimilarityClass[Spc]
               ) -> SimilarityClass[Spc] | NotImplementedType:
        # Handle all inputs that are not SwitchTimeClass.
        if isinstance(b, UniversalClass):
            return b
        if isinstance(b, EmptyClass):
            return self
        if not isinstance(b, SwitchTimeClass):
            return NotImplemented

        # Get switch times from both classes. Ensure that `t_a` has no more
        # entries than t_b.
        t_a, t_b = self._times, b._times
        if len(t_a) > len(t_b):
            t_a, t_b = t_b, t_a

        # Make a copy of t_b.
        t_b = numpy.copy(t_b)

        # Search for insertion points for `a`'s intervals in `b`.
        idx_start = numpy.searchsorted(t_a[::2], t_b, side='left')
        idx_end = numpy.searchsorted(t_a[1::2], t_b, side='right')

        # Iterate over the insertion ranges of `a` in `b` in reverse.
        # NOTE: We do this in reverse so that earlier indices do not shift.
        for i, s, j, t in zip(
            reversed(idx_start), reversed(t_a[::2]),
            reversed(idx_end), reversed(t_a[1::2])
        ):
            if i % 2 == 0:
                comp_left = (t_b[:i], [s])
            else:
                comp_left = (t_b[:i],)
            if j % 2 == 0:
                comp_right = ([t], t_b[j:])
            else:
                comp_right = (t_b[j:],)
            t_b = numpy.concatenate((*comp_left, *comp_right))

        return SwitchTimeClass[Spc](self._space, t_b)

    def __and__(self, b: SimilarityClass[Spc]
                ) -> SimilarityClass[Spc] | NotImplementedType:
        # Handle all inputs that are not SwitchTimeClass.
        if isinstance(b, UniversalClass):
            return self
        if isinstance(b, EmptyClass):
            return b
        if not isinstance(b, SwitchTimeClass):
            return NotImplemented

        # Get switch times from both classes. Ensure that `t_a` has no more
        # entries than t_b.
        t_a, t_b = self._times, b._times
        if len(t_a) > len(t_b):
            t_a, t_b = t_b, t_a

        # Search for insertion points for `a`'s intervals in `b`.
        idx_start = numpy.searchsorted(t_a[::2], t_b, side='right')
        idx_end = numpy.searchsorted(t_a[1::2], t_b, side='left')

        # Iterate over the insertion ranges of `a` in `b` to build component
        # list.
        comp = []
        for i, s, j, t in zip(idx_start, t_a[::2], idx_end, t_a[1::2]):
            if i % 2 != 0:
                comp.append([s])
            comp.append(t_b[i:j])
            if j % 2 != 0:
                comp.append([t])

        return SwitchTimeClass[Spc](self._space, numpy.concatenate(comp))

    def __sub__(self, b: SimilarityClass[Spc]
                ) -> SimilarityClass[Spc] | NotImplementedType:
        # Handle all inputs that are not SwitchTimeClass.
        if isinstance(b, UniversalClass):
            return self._space.empty_class
        if isinstance(b, EmptyClass):
            return self
        if not isinstance(b, SwitchTimeClass):
            return NotImplemented

        # Get switch times from both classes. Ensure that `t_a` has no more
        # entries than t_b.
        t_a, t_b = self._times, b._times

        # Find insertion points of `b` in `a`.
        idx_start = numpy.searchsorted(t_b[::2], t_a, side='right')
        idx_end = numpy.searchsorted(t_b[1::2], t_a, side='left')

        # Iterate over the insertion ranges of `a` in `b` to build component
        # list.
        comp = []
        idx_run = 0
        for i, s, j, t in zip(idx_start, t_b[::2], idx_end, t_b[1::2]):
            comp.append(t_a[idx_run:i])
            if i % 2 != 0:
                comp.append([s])
            if j % 2 != 0:
                comp.append([t])
            idx_run = j
        if idx_run < len(t_a):
            comp.append(t_a[idx_run:])

        return SwitchTimeClass[Spc](self._space, numpy.concatenate(comp))

    def __rsub__(self, a: SimilarityClass[Spc]
                 ) -> SimilarityClass[Spc] | NotImplementedType:
        # Handle all inputs that are not SwitchTimeClass.
        if isinstance(a, UniversalClass):
            return ~self
        if isinstance(a, EmptyClass):
            return a
        if not isinstance(a, SwitchTimeClass):
            return NotImplemented

        # Delegate to __sub__
        return a.__sub__(self)

    def __xor__(self, b: SimilarityClass[Spc]
                ) -> SimilarityClass[Spc] | NotImplementedType:
        # Handle all inputs that are not SwitchTimeClass.
        if isinstance(b, UniversalClass):
            return ~self
        if isinstance(b, EmptyClass):
            return self
        if not isinstance(b, SwitchTimeClass):
            return NotImplemented

        # Merge sorted time vectors.
        #
        # NOTE: This uses a regular sort, though merging of sorted lists can
        # be done more quickly.
        return SwitchTimeClass[Spc](
            self._space, numpy.concatenate((self._times, b._times))
        )
