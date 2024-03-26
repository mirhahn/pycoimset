#!/usr/bin/env python3

# PyCoimset Example "Poisson Design": Main entry point
#
# Copyright 2024 Mirko Hahn
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

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import argparse
import json
import logging
import sys
import warnings

import meshio
import numpy
import skfem

from functionals import MeasureFunctional, ObjectiveFunctional
from pycoimset import PenaltySolver, UnconstrainedSolver
from pycoimset.helpers import with_safety_factor
from pycoimset.helpers.functionals import weighted_sum
from pycoimset.solver.unconstrained.solver import SolverParameters
from space import BoolArrayClass, SimilaritySpace


# Callback for solution recording.
class Callback:
    _tmpl: str
    obj_func: ObjectiveFunctional

    def __init__(self, func: ObjectiveFunctional, file: str = 'iterate_{idx:04d}.vtk'):
        self._tmpl = file
        self.obj_func = func

    def __call__(self, solver: PenaltySolver | UnconstrainedSolver):
        # Extract solution and objective functional.
        sol = solver.x
        if not isinstance(sol, BoolArrayClass):
            return
       
        (obj_func := self.obj_func).arg = sol
        obj_func.get_value()
        obj_func.get_gradient()
        eval = obj_func.evaluator
        
        # Retrieve evaluation mesh
        mesh = obj_func.evaluator.mesh
        
        # Generate meshio mesh.
        mesh = meshio.Mesh(
            numpy.vstack((mesh.p, numpy.broadcast_to(0, (1, mesh.nvertices)))).T,
            cells={
                'triangle': mesh.t.T
            },
            point_data={
                'pdesol': eval.pdesol,
                'adjsol': eval.adjsol
            },
            cell_data={
                'control': [eval.ctrl],
                'grad': [(1 - 2 * eval.ctrl) * (eval.grad / eval.vol)]
            }
        )
        
        # Write to file.
        mesh.write(self._tmpl.format(idx=solver.stats.n_iter))


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-pp', '--problem', dest='problem_parameters',
                    type=argparse.FileType('r'), help='JSON file with problem '
                    'parameters.')
parser.add_argument('-sp', '--solver', dest='solver_parameters',
                    type=argparse.FileType('r'), help='JSON file with solver '
                    'parameters.')
parser.add_argument('-p', '--param', dest='parameters',
                    type=argparse.FileType('r'), help='Combined JSON file '
                    'with problem and solver parameters.')
parser.add_argument('-op', '--param-out', dest='parameter_output',
                    type=argparse.FileType('w'), help='Write combined JSON '
                    'file with problem and solver parameters.')
parser.add_argument('-o', '--output', type=str,
                    default='iterate_{idx:04d}.vtk',
                    help='Template for output paths')
parser.add_argument('-v', '--verbose', nargs='?', type=int, default=0,
                    help='Include debugging output (optional level).')
args = parser.parse_args()

# Set up logging.
logging.basicConfig(format=logging.BASIC_FORMAT, stream=sys.stdout)
logging.getLogger('skfem').setLevel(logging.ERROR)
if args.verbose <= 0:
    logging.getLogger('pycoimset').setLevel(logging.WARNING)
    logging.getLogger('space').setLevel(logging.INFO)
    logging.getLogger('pde.evaluator').setLevel(logging.INFO)
elif args.verbose <= 1:
    logging.getLogger('pycoimset').setLevel(logging.INFO)
    logging.getLogger('space').setLevel(logging.DEBUG)
    logging.getLogger('pde.evaluator').setLevel(logging.DEBUG)
else:
    logging.getLogger('pycoimset').setLevel(logging.DEBUG)
    logging.getLogger('space').setLevel(logging.DEBUG)
    logging.getLogger('pde.evaluator').setLevel(logging.DEBUG)

# Set resource limits and convert warnings to exceptions.
#resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.simplefilter('error')

# Extract parameters.
sol_param = {}
prob_param = {}
if args.parameters is not None:
    with args.parameters as f:
        d = json.load(f)
    assert isinstance(d, dict)
    sol_param = d.get('solver', sol_param)
    prob_param = d.get('problem', prob_param)
if args.solver_parameters is not None:
    with args.solver_parameters as f:
        d = json.load(f)
    assert isinstance(d, dict)
    sol_param = d
if args.problem_parameters is not None:
    with args.problem_parameters as f:
        d = json.load(f)
    assert isinstance(d, dict)
    prob_param = d
assert isinstance(sol_param, dict)
assert isinstance(prob_param, dict)

# Construct initial mesh.
initial_mesh = skfem.MeshTri().refined(prob_param.get('initial_resolution', 6))
assert isinstance(initial_mesh, skfem.MeshTri)
space = SimilaritySpace(initial_mesh)
ctrl = BoolArrayClass(space, space.mesh)

# Set up solver.
sol_type = sol_param.pop('type', 'penalty')
if sol_type == 'unconstrained':
    sol_param = SolverParameters(**sol_param)
    obj = ObjectiveFunctional(space)
    solver = UnconstrainedSolver(
        weighted_sum(
            [
                with_safety_factor(
                    obj,
                    prob_param.setdefault('safety_factor', 0.05)
                ),
                MeasureFunctional(space)
            ],
            [1.0, prob_param.setdefault('mu_init', 8.75e-5)],
            [1.0, 0.0],
            [1.0, 0.0]
        ),
        initial_sol=ctrl,
        callback=Callback(obj, args.output),
        param=sol_param
    )
elif sol_type == 'penalty':
    sol_param = PenaltySolver.Parameters(**sol_param)
    obj = ObjectiveFunctional(space)

    # NOTE: With PenaltySolver, objective comes last. Hence, err_wgt is reversed.
    solver = PenaltySolver(
        with_safety_factor(obj,
                           prob_param.setdefault('safety_factor', 0.05)),
        MeasureFunctional(space) <= prob_param.setdefault('measure_bound', 0.4),
        x0=ctrl,
        mu=prob_param.setdefault('mu_init', 0.01),
        err_wgt=[0.0, 1.0],
        param=sol_param,
        callback=Callback(obj, args.output)
    )
else:
    raise ValueError(f'unknown solver type {sol_type}')

# Write parameters if requested.
if args.parameter_output is not None:
    d = {
        'solver': {'type': sol_type},
        'problem': prob_param
    }
    d['solver'].update(sol_param.toJSON())

    with args.parameter_output as f:
        json.dump(d, f, indent=4)

# Solve the problem.
solver.solve()
