#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import logging
import resource
import warnings

import numpy
import skfem

from functionals import MeasureFunctional, ObjectiveFunctional
from pycoimset.solver.unconstrained import SolverParameters
from space import SimilaritySpace, BoolArrayClass

from pycoimset import MOMSolver

logging.disable(logging.WARNING)
resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.simplefilter('error')

initial_mesh = skfem.MeshTri().refined(4)
assert isinstance(initial_mesh, skfem.MeshTri)

space = SimilaritySpace(initial_mesh)
ctrl = BoolArrayClass(space, space.mesh)

solver = MOMSolver(
    ObjectiveFunctional(space),
    [MeasureFunctional(space) <= 0.1],
    initial_solution=ctrl,
    params=SolverParameters(abstol=1e-4, thres_accept=0.1)
)
solver.solve()
