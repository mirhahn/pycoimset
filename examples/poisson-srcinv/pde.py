'''
Implementation of the Poisson topology design problem.

This is primarily based on the example problem "Topology optimisation
of heat conduction problems governed by the Poisson equation" for the
`dolfin-adjoint` package. The problem is originally adapted from
test problem 1 in

    Gersborg-Hansen, A., Bendsoe, M.P. & Sigmund, O. Topology
    optimization of heat conduction problems using the finite volume
    method. Struct Multidisc Optim 31, 251–259 (2006).
    https://doi.org/10.1007/s00158-005-0584-3

Error estimators are heavily based on the dual-weighted residual method
as described in

    Becker, R., & Rannacher, R. (2001). An optimal control approach to
    a posteriori error estimation in finite element methods. Acta
    Numerica, 10, 1-102. doi:10.1017/S0962492901000010

'''

from dataclasses import dataclass, field
from functools import cached_property
import math
import time
from typing import NamedTuple, Optional, Union, cast

import numpy
from numpy.typing import ArrayLike
import scipy.sparse
import scipy.sparse.linalg
import skfem
from skfem.helpers import dot, grad

from util import notify_property_update, tracks_dependencies, depends_on


def k(w):
    return 0.01 + 0.99 * w


def dkdw(d):
    return 0.99 * d


def poisson_system(spc_sol: skfem.Basis, spc_ctrl: skfem.Basis,
                   ctrl: numpy.ndarray, src_dens: float, adjoint: bool = False
                   ) -> skfem.CondensedSystem:
    # Get mesh.
    mesh = spc_sol.mesh
    assert spc_ctrl.mesh is mesh

    # Define the remap function.
    if adjoint:
        remap = lambda u, v: (v, u)
    else:
        remap = lambda u, v: (u, v)

    # Assemble cell integral.
    @skfem.BilinearForm
    def bilinear_interior_form(u, v, w):
        u, v = remap(u, v)
        gu = cast(numpy.ndarray, grad(u))
        gv = cast(numpy.ndarray, grad(v))
        return dot(k(w.p) * gu, gv)
    a_mat = bilinear_interior_form.assemble(
        spc_sol, p=spc_sol.with_element(spc_ctrl.elem()).interpolate(ctrl)
    )

    # Assemble weight vector.
    @skfem.LinearForm
    def linear_form(v, w):
        return w.f * v
    b_vec = linear_form.assemble(spc_sol, f=src_dens)

    # Condense system.
    return skfem.condense(a_mat, b_vec, D=spc_sol.get_dofs(['top', 'left']))


def poisson_control_deriv(spc_sol: skfem.Basis, spc_ctrl: skfem.Basis,
                          ctrl: numpy.ndarray, pde_sol: numpy.ndarray
                          ) -> scipy.sparse.spmatrix:
    # Get mesh.
    mesh = spc_sol.mesh
    assert spc_ctrl.mesh is mesh

    # Assemble cell integral.
    @skfem.BilinearForm
    def interior_form(u, v, w):
        return dot(dkdw(v) * grad(w.y), grad(u))    # pyright: ignore
    a_mat = interior_form.assemble(
        spc_sol, spc_ctrl, y=spc_sol.interpolate(pde_sol)
    )

    return a_mat


def interpolate_local_p2(spc_sol: skfem.Basis, dof_sol: numpy.ndarray,
                         spc_ctrl: skfem.Basis, dof_ctrl: numpy.ndarray,
                         src_dens: float
                         ) -> numpy.ndarray:
    # Set up discontinuous space.
    mesh = spc_sol.mesh
    p2_dg = skfem.Basis(mesh, skfem.ElementDG(skfem.ElementTriP2()))

    # Set up system.
    @skfem.BilinearForm
    def bilin(u, v, w):
        p = w.p
        gu = cast(numpy.ndarray, grad(u))
        gv = cast(numpy.ndarray, grad(v))
        return dot(k(p) * gu, gv)
    @skfem.LinearForm
    def lin(v, w):
        return w.f * v

    a_mat = bilin.assemble(p2_dg, p=p2_dg.with_element(spc_ctrl.elem()).interpolate(dof_ctrl))
    b_vec = lin.assemble(p2_dg, f=src_dens)

    # Identify nodal values for condensation.
    p2dg_nodal_dofs = p2_dg.element_dofs[:3].flatten('F')
    p1_nodal_dofs = dof_sol[spc_sol.nodal_dofs].flatten()
    p2dg_nodal_values = p1_nodal_dofs[mesh.t].flatten('F')

    # Solve.
    x0 = numpy.zeros(p2_dg.N)
    x0[p2dg_nodal_dofs] = p2dg_nodal_values
    x = skfem.solve(*skfem.condense(a_mat, b_vec, D=p2dg_nodal_dofs, x=x0))

    return x


class FunctionSpaces:
    p0: skfem.Basis
    p1: skfem.Basis
    p2: skfem.Basis
    p2_dg: skfem.Basis

    def __init__(self, mesh: skfem.MeshTri):
        self.p0 = skfem.Basis(mesh, skfem.ElementTriP0())
        self.p1 = skfem.Basis(mesh, skfem.ElementTriP1())
        self.p2 = skfem.Basis(mesh, skfem.ElementTriP2())
        self.p2_dg = skfem.Basis(mesh, skfem.ElementDG(skfem.ElementTriP2()))


@tracks_dependencies
class AssembledForms:
    _spc: FunctionSpaces

    _k: Optional[numpy.ndarray]
    _f: Optional[float]
    _y: Optional[numpy.ndarray]

    def __init__(self, spaces: FunctionSpaces):
        self._spc = spaces

        self._k = None
        self._f = None
        self._y = None

    @property
    def spaces(self) -> FunctionSpaces:
        '''Basic function spaces.'''
        return self._spc

    @property
    def k(self) -> Optional[numpy.ndarray]:
        '''Diffusivity parameter.'''
        return self._k

    @k.setter
    def k(self, val: Optional[numpy.ndarray]):
        self._k = val
        notify_property_update(self, 'k')

    def get_k(self) -> numpy.ndarray:
        '''Get diffusivity parameter or throw exception.'''
        assert self._k is not None
        return self._k

    @property
    def f(self) -> Optional[float]:
        '''Source density parameter.'''
        return self._f

    @f.setter
    def f(self, val: Optional[float]):
        self._f = val
        notify_property_update(self, 'f')

    def get_f(self) -> float:
        '''Get source density parameter or raise exception.'''
        assert self._f is not None
        return self._f

    @property
    def y(self) -> Optional[numpy.ndarray]:
        '''Current solution.'''
        return self._y

    @y.setter
    def y(self, val: Optional[numpy.ndarray]):
        self._y = val
        notify_property_update(self, 'y')

    def get_y(self) -> numpy.ndarray:
        '''Get current solution or raise exception.'''
        assert self._y is not None
        return self._y

    @depends_on(k, f)
    @cached_property
    def fwdsys(self) -> skfem.CondensedSystem:
        '''Assembled Poisson system.'''
        s = self._spc
        return poisson_system(s.p1, s.p0, self.get_k(), self.get_f())

    @depends_on(k, f)
    @cached_property
    def fwdsys_high(self) -> skfem.CondensedSystem:
        '''Assembled Poisson system.'''
        s = self._spc
        return poisson_system(s.p2, s.p0, self.get_k(), self.get_f())

    @depends_on(y, f)
    @cached_property
    def objval(self) -> float:
        '''Objective function value.'''
        @skfem.Functional
        def objective(w):
            return w.f * w.y
        s = self._spc
        return objective.assemble(s.p1, f=self.get_f(),
                                  y=s.p1.interpolate(self.get_y()))

    @depends_on(k, f)
    @cached_property
    def adjsys(self) -> skfem.CondensedSystem:
        '''Assembled adjoint system.'''
        s = self._spc
        return poisson_system(s.p1, s.p0, self.get_k(), self.get_f(),
                              adjoint=True)

    @depends_on(k, f)
    @cached_property
    def adjsys_high(self) -> skfem.CondensedSystem:
        '''Assembled higher-order adjoint system.'''
        s = self._spc
        return poisson_system(s.p2, s.p0, self.get_k(), self.get_f(),
                              adjoint=True)

    @depends_on(y)
    @cached_property
    def gradadj2ctrl(self) -> scipy.sparse.spmatrix:
        '''Assembled adjoint-to-control gradient mapping.'''
        s = self._spc
        return poisson_control_deriv(s.p1, s.p0, self.get_k(), self.get_y())


@dataclass
class EvaluationStatistic:
    #: Number of evaluations.
    num: int = 0

    #: Runtime of evaluations.
    time: float = 0.0

    @property
    def avg_time(self) -> float:
        '''Average evaluation time.'''
        return self.time / self.num


@tracks_dependencies
class PoissonEvaluator:
    '''
    Objective functional for the Poisson topology design.
    '''
    class Tolerances(NamedTuple):
        obj: float
        grad: float

    @dataclass
    class Statistics:
        #: PDE solve statistic.
        pdesol: EvaluationStatistic = field(
            default_factory=EvaluationStatistic
        )

        #: PDE solve statistic.
        qpdesol: EvaluationStatistic = field(
            default_factory=EvaluationStatistic
        )

        #: Adjoint solve statistic.
        adjsol: EvaluationStatistic = field(
            default_factory=EvaluationStatistic
        )

        #: Adjoint solve statistic.
        qadjsol: EvaluationStatistic = field(
            default_factory=EvaluationStatistic
        )

        #: Objective function evaluation statistic.
        obj: EvaluationStatistic = field(default_factory=EvaluationStatistic)

        #: Gradient evaluation statistic.
        grad: EvaluationStatistic = field(default_factory=EvaluationStatistic)

    _mesh: skfem.MeshTri
    _forms: AssembledForms
    _tol: Tolerances
    _stats: Statistics

    def __init__(self, mesh: skfem.Mesh, ctrl_dof: numpy.ndarray,
                 obj_tol: float = 1e-6, grad_tol: float = 1e-6):
        '''
        Constructor.

        Parameters
        ----------
        mesh : skfem.Mesh
            Control mesh.
        ctrl_dof : numpy.ndarray
            Control function DOF vector. Must be P0 on control mesh.
        obj_tol : float, optional
            Tolerance for objective evaluation. Defaults to `1e-6`.
        grad_tol : float, optional
            Tolerance for gradient evaluation. Defaults to `1e-6`.
        '''
        assert isinstance(mesh, skfem.MeshTri)

        self._mesh = mesh
        spc = FunctionSpaces(self._mesh)
        self._forms = AssembledForms(spc)
        self._forms.k = ctrl_dof
        self._forms.f = 1e-2
        self._tol = PoissonEvaluator.Tolerances(obj_tol, grad_tol)
        self._stats = PoissonEvaluator.Statistics()

    @property
    def stats(self) -> 'PoissonEvaluator.Statistics':
        '''Execution statistics.'''
        return self._stats

    @property
    def mesh(self) -> skfem.Mesh:
        '''Control mesh.'''
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: skfem.MeshTri) -> None:
        # Mark relevant boundaries.
        mesh = mesh.with_boundaries({
            'left': lambda x: numpy.isclose(x[0], 0.0),
            'top': lambda x: numpy.isclose(x[1], 1.0),
            'right': lambda x: numpy.isclose(x[0], 1.0),
            'bottom': lambda x: numpy.isclose(x[1], 0.0)
        })

        # Create a new forms object.
        old_forms = self._forms
        spaces = FunctionSpaces(mesh)
        self._forms = AssembledForms(spaces)
        self._forms.f = old_forms.f

        if old_forms.k is not None:
            # Batched projection procedure for P0
            x_center = mesh.mapping().F(numpy.array([[1/3], [1/3]])).squeeze(-1)
            finder = self._mesh.element_finder()
            idx_old = numpy.concatenate([finder(*x) for x in numpy.array_split(x_center, math.ceil(x_center.shape[1] / 100), axis=1)])
            self._forms.k = old_forms.k[idx_old]
        del old_forms

        # Replace old mesh.
        self._mesh = mesh

        # Reset everything.
        notify_property_update(self, 'mesh')

    @depends_on(mesh)
    @cached_property
    def vol(self) -> numpy.ndarray:
        '''Cell volumes.'''
        @skfem.Functional
        def vol(_):
            return 1.0
        return vol.elemental(self.spaces.p0)

    @property
    def spaces(self) -> FunctionSpaces:
        '''Function spaces.'''
        return self._forms.spaces

    @property
    def ctrl(self) -> numpy.ndarray:
        '''DOF vector of the control function.'''
        return self._forms.get_k()

    @property
    def tol(self) -> 'PoissonEvaluator.Tolerances':
        '''Evaluation tolerances.'''
        return self._tol

    @depends_on(mesh)
    @cached_property
    def pdesol(self) -> numpy.ndarray:
        '''DOF vector of the PDE solution.'''
        time_start = time.perf_counter()
        result = cast(
            numpy.ndarray,
            skfem.solve(*self._forms.fwdsys)    # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.pdesol.time += time_end - time_start
        self._stats.pdesol.num += 1
        return result

    @depends_on(mesh)
    @cached_property
    def qpdesol(self) -> numpy.ndarray:
        '''DOF vector of the PDE solution.'''
        time_start = time.perf_counter()
        result = cast(
            numpy.ndarray,
            skfem.solve(*self._forms.fwdsys_high)    # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.qpdesol.time += time_end - time_start
        self._stats.qpdesol.num += 1
        return result

    @depends_on(pdesol)
    @cached_property
    def obj(self) -> float:
        '''Objective function value.'''
        if self._forms.y is not self.pdesol:
            self._forms.y = self.pdesol
        time_start = time.perf_counter()
        result = self._forms.objval
        time_end = time.perf_counter()
        self._stats.obj.time += time_end - time_start
        self._stats.obj.num += 1
        return result

    @depends_on(mesh)
    @cached_property
    def adjsol(self) -> numpy.ndarray:
        '''Adjoint solution DOF vector.'''
        time_start = time.perf_counter()
        result = cast(
            numpy.ndarray,
            skfem.solve(*self._forms.adjsys)    # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.adjsol.time += time_end - time_start
        self._stats.adjsol.num += 1
        return result

    @depends_on(mesh)
    @cached_property
    def qadjsol(self) -> numpy.ndarray:
        '''Higher-order adjoint solution DOF vector.'''
        time_start = time.perf_counter()
        result = cast(
            numpy.ndarray,
            skfem.solve(*self._forms.adjsys_high)   # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.qadjsol.time += time_end - time_start
        self._stats.qadjsol.num += 1
        return result

    @depends_on(adjsol, vol)
    @cached_property
    def grad(self) -> numpy.ndarray:
        '''Gradient DOF vector.'''
        if self._forms.y is not self.pdesol:
            self._forms.y = self.pdesol
        adj_sol = self.adjsol
        time_start = time.perf_counter()
        result = -self._forms.gradadj2ctrl.dot(adj_sol)
        time_end = time.perf_counter()
        self._stats.grad.time += time_end - time_start
        self._stats.grad.num += 1
        return result

    @depends_on(mesh, qadjsol, pdesol)
    @cached_property
    def objerr(self) -> numpy.ndarray:
        '''Cellwise representation of objective error.'''
        # Ensure that solution information is present.
        if self._forms.y is None:
            self._forms.y = self.pdesol

        # Retrieve quadrature bases.
        s = self._forms.spaces
        m = self._mesh

        # Project higher-order adjoint solution onto P1.
        f = self._forms.f
        z = self.qadjsol
        zh = cast(numpy.ndarray, s.p1.project(s.p1.with_element(s.p2.elem()).interpolate(z)))

        # Interior residual.
        @skfem.Functional
        def interior_residual(w):
            return w.f * (w.z - w.zh)
        eta_int = interior_residual.elemental(
            s.p2, f=f, z=z,
            zh=s.p2.with_element(s.p1.elem()).interpolate(zh)
        )

        # Set up buffer for facet terms.
        eta_fac = numpy.zeros(m.nfacets)

        # Create facet bases.
        p2_int = [skfem.InteriorFacetBasis(self.mesh, s.p2.elem(), side=i) for i in range(2)]
        p1_int = [skfem.InteriorFacetBasis(self.mesh, s.p1.elem(), side=i, quadrature=q.quadrature) for i, q in enumerate(p2_int)]
        p0_int = [skfem.InteriorFacetBasis(self.mesh, s.p0.elem(), side=i, quadrature=q.quadrature) for i, q in enumerate(p2_int)]
        p2_bnd = skfem.FacetBasis(self.mesh, s.p2.elem(), side=0, facets=['right', 'bottom'])
        p1_bnd = skfem.FacetBasis(self.mesh, s.p1.elem(), side=0, facets=['right', 'bottom'], quadrature=p2_bnd.quadrature)
        p0_bnd = skfem.FacetBasis(self.mesh, s.p0.elem(), side=0, facets=['right', 'bottom'], quadrature=p2_bnd.quadrature)

        # Assemble Neumann facet terms.
        @skfem.Functional
        def neumann_facet_residual(w):
            n, p, y, z, zh = w.n, w.p, w.y, w.z, w.zh
            gy = cast(numpy.ndarray, grad(y))
            return dot(k(p) * gy, n) * (z - zh)
        eta_fac[p2_bnd.find] = neumann_facet_residual.elemental(
            p2_bnd,
            p=p0_bnd.interpolate(self._forms.get_k()),
            y=p1_bnd.interpolate(self._forms.get_y()),
            z=p2_bnd.interpolate(z),
            zh=p1_bnd.interpolate(zh)
        )

        # Assemble interior facet terms.
        @skfem.Functional
        def interior_facet_residual(w):
            n, p, y, z, zh = w.n, w.p, w.y, w.z, w.zh
            gy = cast(numpy.ndarray, grad(y))
            return (-1)**idx * dot(k(p) * gy, n) * (z - zh)
        for idx, (p0, p1, p2) in enumerate(zip(p0_int, p1_int, p2_int)):
            eta_fac[p2.find] += interior_facet_residual.elemental(
                p2,
                p=p0.interpolate(self._forms.get_k()),
                y=p1.interpolate(self._forms.get_y()),
                z=p2.interpolate(z),
                zh=p1.interpolate(zh)
            )

        # Map per-facet quantities to their incident cells.
        eta_bnd = numpy.sum(eta_fac[m.t2f], axis=0) / 2

        return eta_int - eta_bnd


    @depends_on(mesh, pdesol, qpdesol, adjsol, qadjsol)
    @cached_property
    def graderr(self) -> numpy.ndarray:
        '''Gradient L^1 error estimator.'''
        # Retrieve spaces and mesh.
        s = self.spaces
        m = self.mesh

        # Interpolate higher order approximations.
        qadjsol = interpolate_local_p2(s.p1, self.adjsol, s.p0, self._forms.get_k(), self._forms.get_f())
        qpdesol = interpolate_local_p2(s.p1, self.pdesol, s.p0, self._forms.get_k(), self._forms.get_f())

        # Define interior residual functional.
        @skfem.Functional
        def interior_residual(w):
            y, yh, z, zh = w.y, w.yh, w.z, w.zh
            gy, gyh, gz, gzh = (
                cast(numpy.ndarray, grad(f)) for f in (y, yh, z, zh)
            )
            return dkdw(dot(gy - gyh, gz - gzh))
        eta_int = interior_residual.elemental(
            s.p2_dg,
            y=s.p2_dg.interpolate(qpdesol),
            z=s.p2_dg.interpolate(qadjsol),
            yh=s.p2_dg.with_element(s.p1.elem()).interpolate(self.pdesol),
            zh=s.p2_dg.with_element(s.p1.elem()).interpolate(self.adjsol),
            p=s.p2_dg.with_element(s.p0.elem()).interpolate(self._forms.get_k())
        )

        # Create buffer for facet terms.
        fac_buf = numpy.zeros((2, m.nfacets))

        # Assemble facet terms.
        p2s = [skfem.InteriorFacetBasis(m, s.p2_dg.elem(), side=0),
               skfem.InteriorFacetBasis(m, s.p2_dg.elem(), side=1),
               skfem.FacetBasis(m, s.p2_dg.elem(), side=0, facets=['right', 'bottom'])]
        p1s = [skfem.InteriorFacetBasis(m, s.p1.elem(), side=0, quadrature=p2s[0].quadrature),
               skfem.InteriorFacetBasis(m, s.p1.elem(), side=1, quadrature=p2s[1].quadrature),
               skfem.FacetBasis(m, s.p1.elem(), side=0, facets=['right', 'bottom'], quadrature=p2s[2].quadrature)]
        sides = [0, 1, 0]

        @skfem.Functional
        def facet_residual(w):
            idx, n, y, z, zh = w.idx, w.n, w.y, w.z, w.zh
            gy = cast(numpy.ndarray, grad(y))
            return (-1)**idx * dkdw(dot(gy, n) * (z - zh))
        for p2, p1, side in zip(p2s, p1s, sides):
            fac_buf[side, p2.find] = facet_residual.elemental(
                p2,
                idx=side,
                y=p1.interpolate(self.adjsol),
                z=p2.interpolate(qpdesol),
                zh=p1.interpolate(self.pdesol)
            ) + facet_residual.elemental(
                p2,
                idx=side,
                y=p1.interpolate(self.pdesol),
                z=p2.interpolate(qadjsol),
                zh=p1.interpolate(self.adjsol)
            )

        numpy.add.at(eta_int, m.f2t, fac_buf)
        return eta_int

    def eval_obj(self) -> float:
        '''
        Evaluate objective to given tolerance.
        '''
        while (err := abs(numpy.sum((eta := self.objerr)))) > self._tol.obj:
            eta = numpy.abs(eta) * self.mesh.param()**2
            sort_idx = numpy.argsort(eta)
            cum_err = numpy.cumsum(eta[sort_idx])
            split_idx = numpy.searchsorted(cum_err, 0.9 * cum_err[-1])
            if split_idx == cum_err.size:
                split_idx = cum_err.size - 1
            where = sort_idx[split_idx:]

            print(f'Objective Error = {err}; refining {where.size}/{self._mesh.nelements} cells')

            self.mesh = self._mesh.refined(where)

        print(f'Objective Error = {err}')
        return self.obj

    def eval_grad(self) -> float:
        '''
        Evaluate objective to given tolerance.
        '''
        while (err := abs(numpy.sum((eta := self.graderr)))) > self._tol.grad:
            eta = numpy.abs(eta) * self.mesh.param()**2
            sort_idx = numpy.argsort(eta)
            cum_err = numpy.cumsum(eta[sort_idx])
            split_idx = numpy.searchsorted(cum_err, 0.9 * cum_err[-1])
            if split_idx == cum_err.size:
                split_idx = cum_err.size - 1
            where = sort_idx[split_idx:]

            print(f'Gradient Error = {err}; refining {where.size}/{self._mesh.nelements} cells')

            self.mesh = self._mesh.refined(where)

        print(f'Gradient Error = {err}')
        return self.grad
