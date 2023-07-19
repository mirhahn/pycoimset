#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import resource
from typing import cast
import warnings

import numpy
import skfem

from pde import PoissonEvaluator

resource.setrlimit(resource.RLIMIT_DATA, (2 * 2**30, 3 * 2**30))
warnings.filterwarnings('error')

mesh = skfem.MeshTri().refined(3)
basis_ctrl = skfem.Basis(mesh, skfem.ElementTriP0())
k = cast(numpy.ndarray, basis_ctrl.project(lambda x: numpy.where(x[0] + x[1] >= 1, 1.0, 0.0)))
eval = PoissonEvaluator(mesh, k)
eval.eval_obj()
eval.eval_grad()

x = eval.pdesol
qx = eval.qpdesol
z = eval.adjsol
qz = eval.qadjsol
g = (1 - 2 * eval._forms.get_k()) * eval.grad / eval.vol
f = eval.obj
ef = eval.objerr
eg = eval.graderr

print(f'Objective value: {f}')
print(f'Objective error: {abs(numpy.sum(ef))}')
print(f'Gradient error: {abs(numpy.sum(eg))}')

for val, basis_or_mesh, name, kwargs in (
    (k, basis_ctrl, 'Control', {}),
    (x, eval.spaces.p1, 'State', {'shading': 'gouraud'}),
    (z, eval.spaces.p1, 'Adjoint State', {'shading': 'gouraud'}),
    (g, eval.spaces.p0, 'Gradient', {}),
    (ef, eval.mesh, 'Objective Error', {}),
    (eg, eval.mesh, 'Gradient Error', {}),
):
    ax = basis_or_mesh.plot(val, colorbar=True, nref=0, **kwargs)
    ax = eval.mesh.draw(ax=ax, plot_kwargs={'alpha': 0.25})
    ax = mesh.draw(ax=ax)
    ax.set_title(name)
    ax.show()
