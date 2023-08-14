#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import logging
import resource
import warnings

from pycoimset import PenaltySolver
from pycoimset.helpers import transform, with_safety_factor
import skfem

from functionals import MeasureFunctional, ObjectiveFunctional
from space import BoolArrayClass, SimilaritySpace

# Set up logging.
logging.basicConfig(format=logging.BASIC_FORMAT)
logging.getLogger('pycoimset').setLevel(logging.INFO)
logging.getLogger('skfem').setLevel(logging.ERROR)
logging.getLogger('space').setLevel(logging.DEBUG)

# Set resource limits and convert warnings to exceptions.
resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.simplefilter('error')

initial_mesh = skfem.MeshTri().refined(5)
assert isinstance(initial_mesh, skfem.MeshTri)

space = SimilaritySpace(initial_mesh)
ctrl = BoolArrayClass(space, space.mesh)

sol_param = PenaltySolver.Parameters(
    abstol=1e-5,
    feas_tol=1e-3,
    thres_accept=0.1,
    thres_reject=0.3,
    thres_tr_expand=0.7,
    margin_instat=0.1,
    margin_proj_desc=0.1,
    margin_step=0.25
)
solver = PenaltySolver(
    with_safety_factor(ObjectiveFunctional(space), 2.0),
    MeasureFunctional(space) <= 0.4,
    x0=ctrl,
    err_wgt=[1.0, 0.0],
    param=sol_param
)
solver.solve()
