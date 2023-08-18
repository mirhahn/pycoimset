#!/usr/bin/env python3

# PyCoimset Example "Lotka-Volterra": Main entry point
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
from typing import cast

from pycoimset import UnconstrainedSolver
from pycoimset.helpers import with_safety_factor
from pycoimset.solver.unconstrained import SolverStats, SolverStatus

import lotka_volterra.ext.scipy as scipy_ext
from lotka_volterra.objective import LotkaObjectiveFunctional
from lotka_volterra.space import IntervalSimilarityClass, IntervalSimilaritySpace


# Treat warnings as errors.
__import__('warnings').simplefilter('error')

# Solver status messages.
StatusMessages = {
    SolverStatus.RUNNING: "Solver is still running.",
    SolverStatus.STATIONARY: "Terminated in epsilon-stationary point.",
    SolverStatus.ERROR_UNKNOWN: "An unknown error occurred.",
    SolverStatus.ERROR_MAX_ITER: "Maximum iteration count reached.",
    SolverStatus.ERROR_PRECISION: "Required precision too high.",
    SolverStatus.ERROR_INTERRUPTED: "User interruption.",
}

# Register interpolant derivative extensions for SciPy.
scipy_ext.register_extensions()

# Define optimization problem
space = IntervalSimilaritySpace((0.0, 12.0))
objective = with_safety_factor(LotkaObjectiveFunctional(space), 2.0)

# Set up solver.
solver = UnconstrainedSolver(objective)
solver.solve()

# Print final status message.
status_msg = StatusMessages.get(
    solver.status,
    "Unknown status code."
)
print(f"Terminated: {solver.status}: {status_msg}", flush=True)

# Output solution.
sol = solver.solution
obj = sol.val
instat = sol.instationarity
sol_data = {
    'argument': cast(IntervalSimilarityClass, sol.arg).toJSON(),
    'tolerances': {
        'objective': sol.val_tol,
        'gradient': sol.grad_tol
    },
    'objective': {
        'value': obj.value,
        'error': obj.error
    },
    'instationarity': {
        'value': instat.value,
        'error': instat.error
    },
    'solver_parameters': solver.param.toJSON()
}
with open('solution.json', 'w') as f:
    json.dump(sol_data, f, indent=2)
