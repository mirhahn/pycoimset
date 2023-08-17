#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import logging
import resource
import warnings

import meshio
import numpy
from pycoimset import PenaltySolver, UnconstrainedSolver
from pycoimset.helpers import with_safety_factor
import skfem

from functionals import MeasureFunctional, ObjectiveFunctional
from pycoimset.helpers.functionals import weighted_sum
from pycoimset.solver.unconstrained import SolverParameters
from space import BoolArrayClass, SimilaritySpace


# Callback for solution recording.
class Callback:
    _tmpl: str

    def __init__(self, file: str = 'iterate_{idx:04d}.vtk'):
        self._tmpl = file

    def __call__(self, solver: PenaltySolver):
        # Retrieve solution.
        sol = solver.solution
        if not isinstance(sol, BoolArrayClass):
            return

        # Retrieve objective functional.
        obj_func = solver.objective_functional
        if not isinstance(obj_func, with_safety_factor):
            return
        obj_func = obj_func.base_functional
        if not isinstance(obj_func, ObjectiveFunctional):
            return
        
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


# Set up logging.
logging.basicConfig(format=logging.BASIC_FORMAT)
logging.getLogger('pycoimset').setLevel(logging.WARNING)
logging.getLogger('skfem').setLevel(logging.ERROR)
logging.getLogger('space').setLevel(logging.DEBUG)

# Set resource limits and convert warnings to exceptions.
#resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.simplefilter('error')

initial_mesh = skfem.MeshTri().refined(6)
assert isinstance(initial_mesh, skfem.MeshTri)

space = SimilaritySpace(initial_mesh)
ctrl = BoolArrayClass(space, space.mesh)

#sol_param = SolverParameters(
#    abstol=1e-5,
#    thres_accept=0.1,
#    thres_reject=0.6,
#    thres_tr_expand=0.9,
#    margin_instat=0.5,
#    margin_proj_desc=0.5,
#    margin_step=0.5,
#    tr_radius=0.01
#)
#solver = UnconstrainedSolver(
#    weighted_sum(
#        [
#            with_safety_factor(ObjectiveFunctional(space), 2.0),
#            MeasureFunctional(space)
#        ],
#        [1.0, 8.75e-5],
#        [1.0, 0.0],
#        [1.0, 0.0]
#    ),
#    initial_sol=ctrl,
#    callback=Callback(),
#    param=sol_param
#)
#solver.solve()

sol_param = PenaltySolver.Parameters(
    abstol=1e-5,
    feas_tol=1e-5,
    thres_accept=0.1,
    thres_reject=0.2,
    thres_tr_expand=0.5,
    margin_instat=1e-3,
    margin_proj_desc=1e-3,
    margin_step=1e-3,
    tr_radius=0.01
)
solver = PenaltySolver(
    with_safety_factor(ObjectiveFunctional(space), 2.0),
    MeasureFunctional(space) <= 0.4,
    x0=ctrl,
    mu=1e-5,
    err_wgt=[1.0, 0.0],
    param=sol_param,
    callback=Callback()
)
solver.solve()
