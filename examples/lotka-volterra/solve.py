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

import argparse
import json
import logging
import os
import sys
from typing import cast

from pycoimset import UnconstrainedSolver
from pycoimset.helpers import with_safety_factor
from pycoimset.solver.unconstrained import SolverParameters, SolverStats, SolverStatus

import lotka_volterra.ext.scipy as scipy_ext
from lotka_volterra.objective import LotkaObjectiveFunctional
from lotka_volterra.space import IntervalSimilarityClass, IntervalSimilaritySpace


class Callback:
    path_tmpl: str

    def __init__(self, path_tmpl: str):
        self.path_tmpl = path_tmpl

    def __call__(self, solver: UnconstrainedSolver) -> None:
        '''
        Output current solution to a file.
        '''
        # Assemble solution in a JSON-suitable format.
        sol = solver.solution
        n_iter = solver.stats.n_iter
        obj = sol.val
        instat = sol.instationarity
        sol_data = {
            'iteration': n_iter,
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

        # Write to file.
        path = self.path_tmpl.format(i=n_iter)
        with open(path, 'w') as f:
            json.dump(sol_data, f, indent=4)


# Treat warnings as errors.
__import__('warnings').simplefilter('error')


# Register interpolant derivative extensions for SciPy.
scipy_ext.register_extensions()

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--param', type=argparse.FileType('r'),
                    help='JSON file with solver parameters.')
parser.add_argument('-op', '--param-out', type=argparse.FileType('w'),
                    help='Write solver parameters to JSON file')
parser.add_argument('-o', '--output', type=str,
                    default='iterate_{i:04d}.json', help='')
parser.add_argument('-v', '--verbose', type=int, nargs='?', const=1,
                    help='Write debugging output.')
args = parser.parse_args()

# Set up logging.
logging.basicConfig(stream=sys.stdout, format=logging.BASIC_FORMAT)
if args.verbose is not None:
    if args.verbose >= 2:
        logging.getLogger('pycoimset').setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger('pycoimset').setLevel(logging.INFO)

# Obtain solver parameters.
sol_param = {
    "abstol": 1e-3,
    "thres_accept": 0.2,
    "thres_reject": 0.4,
    "thres_tr_expand": 0.6,
    "margin_step": 1e-3,
    "margin_proj_desc": 0.1,
    "margin_instat": 0.5,
    "max_iter": None
}
if args.param is not None:
    with args.param as f:
        sol_param = json.load(f)
    assert isinstance(sol_param, dict)
sol_param = SolverParameters(**sol_param)

# Write parameters if requested.
if args.param_out is not None:
    with args.param_out as f:
        json.dump(sol_param.toJSON(), f, indent=4)

# Define optimization problem
space = IntervalSimilaritySpace((0.0, 12.0))
objective = with_safety_factor(LotkaObjectiveFunctional(space), 2.0)

# Set up solver.
solver = UnconstrainedSolver(
    objective,
    callback=Callback(args.output),
    param=sol_param
)
solver.solve()
