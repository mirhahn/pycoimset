#!/usr/bin/env python3

'''
Solver for the "Topology optimisation of heat conduction problems
governed by the Poisson equation" example of the 
'''

import numpy
import skfem

from pde import PoissonEvaluator

mesh = skfem.MeshTri.init_symmetric().refined(3)
basis_ctrl = skfem.Basis(mesh, skfem.ElementTriP0())
eval = PoissonEvaluator(basis_ctrl, basis_ctrl.zeros())
eval.eval_obj()

x = eval.pdesol
f = eval.obj
g = eval.grad
z = eval.adjsol
qz = eval.qadjsol
err = eval.objerr

print(f)
print(numpy.sum(err))
print(eval.stats)

for val, name, kwargs in (
    (x, 'State', {'shading': 'gouraud'}),
    (z, 'Adjoint State', {'shading': 'gouraud'}),
    (g, 'Gradient', {}),
    (err, 'Error', {}),
):
    ax = eval.mesh.draw()
    eval.mesh.plot(x, ax=ax, colorbar=True, **kwargs)
    ax.set_title(name)
    ax.show()
