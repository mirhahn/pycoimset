# PyCoimset: Python library for COntinuous IMprovement of SETs
# Copyright 2025 Mirko Hahn
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

from typing import cast

import numpy
import pytest

from pycoimset.spaces.interval import IntervalBorelSpace, SwitchTimeClass


# RAW DATA
_switch_times = [
    # unsorted, not deduplicated, between 0 and 1
    [0.25, 0.9375, 0.5625, 0.5625, 0.8125, 0.25, 0.25, 0.0625, 0.625, 0.6875],
    [0.5, 0.125, 0.5, 0.75, 0.0625, 0.0625, 0., 0.25, 0.375],
    [0.375, 0.4375, 0.875, 0.625, 0.875, 0.5, 0.3125, 0.4375, 0.],
    # sorted, not deduplicated, between 0 and 1
    [0.0, 0.1875, 0.5, 0.5625, 0.625, 0.75, 0.8125, 0.875, 0.875, 1.0],
    [0.0, 0.0625, 0.0625, 0.0625, 0.3125, 0.5, 0.6875, 0.6875, 0.8125, 0.875],
    [0.0, 0.4375, 0.5, 0.625, 0.6875, 0.6875, 0.6875],
    # sorted, deduplicated, between 0 and 1
    [0.0625, 0.375, 0.4375, 0.75],
    [0.0, 0.0625, 0.125, 0.1875, 0.5, 0.75],
    [0.0625, 0.75, 0.8125, 0.875, 0.9375, 1.0],
    [0.0, 0.125, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.875, 0.9375, 1.0],
    # unsorted, not deduplicated, between -15 and 15
    [-7.25, -9.25, -6.9375, -0.1875, 6.6875, 8.125, -11.625, -4.875, 4.875,
     -10.8125, -11.375, 8.1875, -4.125, -4.3125, 9.0625, -4.0625, 7.625, -8.75,
     1.5625, 10.375, 7.0625, -10.1875, 13.75, -13.1875, 13.0, 7.25, -11.1875,
     -11.25, 0.875, -1.1875, -6.875, -8.25, -13.625, 9.125, 5.5, -7.4375,
     -10.9375, 7.875, 8.75, -1.6875, -1.0625, 8.0, -8.875, -3.9375, -3.0625,
     -11.125, 3.875, -7.3125, 1.875, 14.375, -10.375, 1.1875, -12.1875, 2.125,
     13.0625, -14.8125, -4.9375, 11.875, -9.5625, -7.875, 1.375, 8.5625,
     3.6875, -4.0625],
    [-6.625, -3.1875, -8.25, 11.8125, -12.0625, -8.125, 1.4375, 13.6875,
     9.9375, -8.0, -11.0, 10.375, 3.0625, 12.75, 10.3125, -5.0625, -0.5, 9.875,
     -6.5, -9.0625, 5.9375, -9.125, 0.5, 3.125, -11.5625, -7.875, -13.4375,
     -5.1875, -1.3125, 11.5625, -9.75, -8.5625, 1.375, -14.0, -3.0625, 1.75,
     -10.0, -5.9375, -2.625, 8.875, 11.9375, -11.4375, 0.1875, 3.4375, -1.9375,
     6.9375, -2.8125, -4.375, -5.3125, 1.4375, -12.3125, 11.875, 8.9375,
     -12.6875, 0.0625, 3.4375, -14.5625, 6.75, -13.6875, 5.875, -3.4375, -6.25,
     -13.0625, -9.0625],
    [12.3125, 2.6875, 3.125, -6.8125, 6.0625, 8.4375, -10.0, -8.0, 0.9375,
     9.25, -4.125, -3.5, 8.0625, 10.375, -13.125, -6.4375, -9.4375, -1.6875,
     -3.375, -0.125, 3.0625, 0.75, -0.3125, 11.1875, -8.4375, -11.4375, -3.75,
     8.5625, -3.6875, 8.4375, -10.1875, -12.9375, -9.4375, 7.6875, -0.9375,
     -2.625, -8.9375, 11.4375, 1.125, 0.3125, -8.75, -5.6875, -1.75, 8.0,
     -3.0625, -0.9375, -0.1875, -6.3125, -3.8125, -1.625, -9.1875, -5.4375,
     11.375, 12.3125, -14.0, -10.625, 1.5625, -7.75, -3.5, 12.9375, 4.3125,
     10.5625, -9.5, -10.0625],
    # sorted, not deduplicated, between -15 and 15
    [-14.6875, -14.25, -13.875, -13.6875, -13.4375, -13.0, -12.6875, -11.875,
     -11.5, -11.3125, -11.3125, -11.1875, -10.75, -10.5625, -10.125, -9.3125,
     -9.1875, -8.8125, -8.75, -8.6875, -8.1875, -7.6875, -6.4375, -5.875,
     -5.625, -4.125, -3.25, -3.0, -2.6875, -1.75, -1.375, -1.25, -0.8125,
     -0.75, -0.3125, -0.1875, 0.0625, 0.0625, 1.5625, 1.75, 1.8125, 3.3125,
     3.3125, 5.4375, 6.125, 6.4375, 6.5, 7.5625, 7.625, 8.1875, 8.25, 9.0, 9.0,
     9.6875, 10.0, 10.4375, 11.0625, 11.125, 11.625, 12.5625, 13.9375, 14.125,
     14.6875, 14.9375],
    [-13.5625, -11.75, -11.5625, -11.375, -11.0, -10.8125, -10.8125, -10.6875,
     -10.4375, -9.375, -9.3125, -8.8125, -8.75, -8.5625, -8.125, -8.0625, -6.0,
     -5.375, -5.3125, -5.25, -5.1875, -2.9375, -2.625, -1.9375, -1.8125,
     -0.8125, 0.25, 0.875, 0.9375, 1.6875, 2.1875, 2.3125, 3.25, 3.625, 3.8125,
     4.0625, 4.5, 4.5, 5.0625, 5.0625, 5.5625, 5.625, 5.8125, 5.875, 6.1875,
     6.4375, 6.5625, 6.625, 6.875, 6.9375, 7.125, 7.3125, 8.4375, 9.4375, 9.5,
     10.5625, 10.75, 10.9375, 11.1875, 12.3125, 12.6875, 12.875, 13.625,
     14.625],
    [-13.3125, -13.0, -12.8125, -12.75, -12.4375, -12.3125, -11.75, -11.25,
     -11.0625, -9.5625, -9.5625, -9.375, -9.0625, -7.9375, -7.875, -7.5625,
     -7.1875, -7.1875, -6.8125, -6.6875, -6.0625, -4.625, -3.9375, -3.4375,
     -3.1875, -2.3125, -0.875, 0.75, 1.8125, 2.25, 2.3125, 3.375, 3.625, 3.625,
     4.75, 4.8125, 5.0625, 5.125, 5.25, 5.5, 6.125, 6.625, 7.25, 7.375, 7.375,
     8.3125, 8.5, 9.125, 9.3125, 9.9375, 9.9375, 10.25, 11.0, 11.3125, 11.375,
     11.625, 11.625, 11.9375, 12.375, 13.375, 13.8125, 13.9375, 14.875, 15.0],
    # sorted, deduplicated, between -15 and 15
    [-14.75, -14.375, -14.25, -13.25, -12.25, -12.125, -12.0, -11.875, -10.625,
     -10.5, -9.625, -9.5, -9.375, -8.75, -7.875, -7.0, -6.25, -6.125, -5.25,
     -4.75, -4.5, -4.25, -3.125, -3.0, -2.75, -2.5, -2.25, -2.125, -1.5,
     -1.375, -1.25, -0.875, -0.625, 0.125, 2.25, 3.0, 3.375, 4.375, 4.625,
     5.125, 5.875, 7.25, 9.375, 9.5, 9.625, 11.0, 11.75, 12.0, 12.125, 12.625,
     13.5, 13.875, 14.375, 14.75],
    [-14.875, -14.375, -14.125, -14.0, -13.125, -12.875, -11.875, -11.625,
     -11.5, -11.125, -10.75, -8.25, -8.0, -7.375, -7.0, -6.375, -6.125, -5.75,
     -4.25, -3.0, -2.5, -1.125, 0.25, 0.375, 1.625, 2.625, 3.5, 4.25, 5.375,
     5.5, 6.25, 6.625, 7.0, 7.125, 7.25, 7.625, 8.0, 8.625, 8.75, 9.0, 9.25,
     10.5, 10.875, 11.125, 11.375, 11.75, 12.0, 12.125, 13.25, 13.875],
    [-14.875, -14.375, -13.25, -12.875, -11.875, -11.75, -10.375, -10.125,
     -9.25, -8.375, -8.25, -7.125, -6.25, -5.5, -5.375, -5.0, -4.75, -4.375,
     -4.125, -3.625, -3.125, -2.625, -1.25, -1.125, 0.0, 1.375, 1.625, 1.75,
     2.875, 4.25, 4.375, 4.75, 6.5, 8.375, 8.5, 8.75, 9.375, 10.375, 11.0,
     12.125, 13.125, 13.75, 13.875, 14.0, 14.5, 14.875]
]


_measure_bounds = [
    (6.5, 5.375),
    (3.125, 1.875),
    (1.625, 18.0),
    (13.75, 16.875),
    (0.0, 0.375),
    (6.125, 14.5),
    (14.375, 8.75),
    (2.125, 14.0),
    (15.75, 7.875),
    (19.5, 18.75),
    (4.5, 1.5),
    (5.5, 8.125),
    (20.0, 9.625),
    (15.625, 9.0),
    (10.125, 6.75),
    (19.5, 2.5),
    (11.375, 17.875),
    (9.5, 5.125),
    (9.25, 19.0),
    (13.0, 18.75),
    (-18.5, 14.0),
    (0.0, 7.625),
    (15.125, -5.5),
    (-0.5, -0.625),
    (1.25, -11.75),
    (-16.125, 18.0),
    (-2.375, -0.625),
    (13.625, -15.0),
    (-14.5, -2.125),
    (-3.625, -7.375),
    (-17.75, 10.75),
    (1.375, -19.0),
    (-5.125, 8.5),
    (0.75, 4.625),
    (-4.5, 12.25),
    (2.125, 4.125),
    (-3.25, -14.625),
    (15.125, -3.0),
    (-9.875, 15.125),
    (15.5, 15.25)
]


# FIXTURES
@pytest.fixture(scope='module', params=[
    {"args": (0, 1), "measure": 1.0, "empty": False},
    {"args": (-10, 10), "measure": 20.0, "empty": False},
    {"args": (10, -10), "measure": 0.0, "empty": True},
])
def space(request: pytest.FixtureRequest):
    return {"space": IntervalBorelSpace(*request.param["args"]),
            **request.param}


@pytest.fixture(scope='module', params=_switch_times)
def switch_times(request):
    return request.param


@pytest.fixture(
    scope='module',
    params=_switch_times
)
def simclass_lhs(space, request):
    return SwitchTimeClass(space["space"], request.param)


@pytest.fixture(
    scope='module',
    params=_switch_times
)
def simclass_rhs(space, request):
    return SwitchTimeClass(space["space"], request.param)


@pytest.fixture(
    scope='module',
    params=_measure_bounds + sorted(_measure_bounds)
)
def measure_bounds(request):
    return request.param


# HELPERS
def check_subset(sub, sup):
    t_sub, t_sup = sub.switch_times, sup.switch_times
    assert all((
        any((
            u <= s and v >= t
            for u, v in zip(t_sup[::2], t_sup[1::2])
        ))
        for s, t in zip(t_sub[::2], t_sub[1::2])
    ))


# TESTS
def test_space_measure(space):
    expected_measure = space["measure"]
    space = space["space"]
    assert space.measure == expected_measure


def test_empty_class(space):
    sim_cls = space["space"].empty_class
    assert sim_cls.measure == 0.0


def test_universal_class(space):
    space = space["space"]
    sim_cls = space.universal_class
    full_measure = space.measure
    assert sim_cls.measure == full_measure


def test_space_empty(space):
    is_empty = space["empty"]
    lb, ub = space["space"].bounds
    assert (ub < lb) == is_empty


def test_switch_time_constructor(space, switch_times):
    space = cast(IntervalBorelSpace, space["space"])
    simcls = SwitchTimeClass(space, switch_times)

    lb, ub = space.bounds
    t = simcls.switch_times

    # Try to artificially slice the correct subsequence out of the input array
    sorted_times = numpy.array(sorted(switch_times), dtype=numpy.float64)
    start = numpy.searchsorted(sorted_times, lb, side='right')
    end = numpy.searchsorted(sorted_times, ub, side='left')

    sliced_times = sorted_times[start:end]
    if start <= end and start % 2 != 0:
        sliced_times = numpy.concatenate(([lb], sliced_times))
    if start <= end and end % 2 != 0:
        sliced_times = numpy.concatenate((sliced_times, [ub]))

    # Verify write protection on switch time array.
    if len(t) > 0:
        with pytest.raises(ValueError, match="read-only"):
            t[:] = 0

    # Test for basic expected properties.
    assert simcls.space is space
    assert len(t) % 2 == 0
    assert all(t <= ub)
    assert all(t >= lb)
    assert all((ts in (lb, ub) or ts in switch_times for ts in t))
    assert all(t[1:] > t[:-1])
    assert numpy.sum(sliced_times[1::2] - sliced_times[::2]) == simcls.measure


def test_simcls_combination(simclass_lhs, simclass_rhs):
    # Check intersection
    simclass_intersect = simclass_lhs & simclass_rhs
    check_subset(simclass_intersect, simclass_lhs)
    check_subset(simclass_intersect, simclass_rhs)

    # Check union
    simclass_union = simclass_lhs | simclass_rhs
    check_subset(simclass_lhs, simclass_union)
    check_subset(simclass_rhs, simclass_union)

    # Check set difference
    simclass_diff_left = simclass_lhs - simclass_rhs
    check_subset(simclass_diff_left, simclass_lhs)
    assert (simclass_diff_left & simclass_rhs).measure == 0

    simclass_diff_right = simclass_rhs - simclass_lhs
    check_subset(simclass_diff_right, simclass_rhs)
    assert (simclass_diff_right & simclass_lhs).measure == 0

    # Check symmetric difference
    simclass_symdiff = simclass_lhs ^ simclass_rhs
    assert numpy.all(
        simclass_symdiff.switch_times
        == (simclass_diff_left | simclass_diff_right).switch_times
    )
    assert (simclass_symdiff & simclass_intersect).measure == 0
    check_subset(simclass_symdiff, simclass_union)

    # Check complement
    simclass_lhs_compl = ~simclass_lhs
    assert (simclass_lhs_compl & simclass_lhs).measure == 0
    check_subset(simclass_diff_right, simclass_lhs_compl)


def test_simcls_subset(simclass_lhs, measure_bounds):
    meas_low, meas_high = measure_bounds

    effective_meas_low = max(meas_low, 0)
    effective_meas_high = min(meas_high, simclass_lhs.measure)
    has_subset = effective_meas_high >= effective_meas_low

    if has_subset:
        simclass_subset = simclass_lhs.subset(meas_low, meas_high)
        check_subset(simclass_subset, simclass_lhs)
        assert meas_low <= simclass_subset.measure
        assert meas_high >= simclass_subset.measure
    else:
        with pytest.raises(ValueError):
            simclass_lhs.subset(meas_low, meas_high)
