#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import logging
import resource
import warnings

import skfem

from functionals import MeasureFunctional, ObjectiveFunctional
from space import SimilaritySpace, BoolArrayClass

from pycoimset import BarrierSolver
from pycoimset.helpers import TransformedFunctional

logging.disable(logging.WARNING)
resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.simplefilter('error')

initial_mesh = skfem.MeshTri().refined(5)
assert isinstance(initial_mesh, skfem.MeshTri)

space = SimilaritySpace(initial_mesh)
ctrl = BoolArrayClass(space, space.mesh)

sol_param = BarrierSolver.Parameters(
    abstol=1e-3,
    compltol=1e-4,
    mu_init=1e-2,
    thres_accept=0.05,
    thres_reject=0.55,
    margin_instat=0.5,
    margin_proj_desc=0.5,
    margin_step=0.25
)
solver = BarrierSolver(
    TransformedFunctional(ObjectiveFunctional(space), scale=10000),
    [MeasureFunctional(space) <= 0.1],
    x0=ctrl,
    err_wgt=[1.0, 0.0],
    param=sol_param
)
solver.solve()
