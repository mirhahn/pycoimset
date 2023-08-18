#!/usr/bin/env python3

# PyCoimset Example "Lotka-Volterra": Plotting script
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

import json
from re import A
from typing import Sequence

from numpy.typing import ArrayLike, NDArray

import lotka_volterra.ext.scipy
from lotka_volterra.objective import LotkaObjectiveFunctional
from lotka_volterra.space import IntervalSimilarityClass
from lotka_volterra.typing import OdeSolutionLike

import numpy
import matplotlib.pyplot as plt


def trajectory_grid(times: ArrayLike, trajectories: Sequence[OdeSolutionLike]
                    ) -> tuple[NDArray, NDArray]:
    '''
    Create a trajectory grid with trajectory piece indices.
    '''
    # Find indices for pre-existing times.
    t_end = numpy.array([tr.t_max for tr in trajectories])
    tr_sorted = numpy.argsort(t_end)
    i_tr = tr_sorted[numpy.searchsorted(t_end, times, side='left')]
    i_tr[i_tr == len(trajectories)] = tr_sorted[-1]

    # Insert switch times.
    t_sw = numpy.empty(2 * len(trajectories) - 2)
    t_sw[::2] = t_end[tr_sorted][:-1]
    t_sw[1::2] = t_end[tr_sorted][:-1]

    i_sw = numpy.empty(2 * len(trajectories) - 2)
    i_sw[::2] = tr_sorted[:-1]
    i_sw[1::2] = tr_sorted[1:]

    # Sort joint grid.
    times = numpy.concatenate((times, t_sw))
    idx = numpy.concatenate((i_tr, i_sw))
    sort_key = numpy.lexsort((idx, times))
    times = times[sort_key]
    idx = idx[sort_key]

    return times, idx


# Load SciPy extensions
lotka_volterra.ext.scipy.register_extensions()

# Load JSON solution
with open('solution.json', 'r') as f:
    sol_dat = json.load(f)

# Create required objects
x = IntervalSimilarityClass.fromJSON(sol_dat['argument'])
f = LotkaObjectiveFunctional(x.space)

# Evaluate functional
f.arg = x
f.val_tol = sol_dat['tolerances']['objective']
f.grad_tol = sol_dat['tolerances']['gradient']
f.get_gradient()

# Generate a sample space
t_sample = numpy.linspace(*x.space.time_range, num=1201, endpoint=True)
t_sample_adj = f.ivp_objects.adj.import_times(t_sample)

# Evaluate primal state and gradient density.
traj_fwd = f.trajectories.fwd
traj_grad = f.trajectories.grad_dens
assert traj_fwd is not None and traj_grad is not None
t_grid, i_grid = trajectory_grid(t_sample, traj_fwd)    # pyright: ignore

y_sample = numpy.empty((len(t_grid), 3))
g_sample = numpy.empty((len(t_grid), 1))
for idx, (y_traj, g_traj) in enumerate(zip(traj_fwd, traj_grad)):
    where = numpy.flatnonzero(i_grid == idx)
    if len(where) > 0:
        y_sample[where, :] = y_traj(t_grid[where]).T
        g_sample[where, :] = g_traj(t_grid[where]) * (-1)**idx

# Generate plots
t_switch = x.switch_times
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

for t0, t1 in zip(t_switch[::2], t_switch[1::2]):
    ax1.axvspan(t0, t1, alpha=0.25)
    ax2.axvspan(t0, t1, alpha=0.25)

ax1.plot(t_grid, y_sample[:, :3], label=('Predator', 'Prey', 'Objective'))
ax1.legend()
ax2.plot(t_grid, numpy.fmin(0.0, g_sample))

ax1.set_title('Primal State')
ax2.set_title('Negative Gradient Density')
plt.show()


