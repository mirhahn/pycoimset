'''
Evaluator for the Poisson problem.
'''

from dataclasses import dataclass, field
from functools import cached_property
import math
import time
from typing import NamedTuple, Optional, cast
from matplotlib.patheffects import withTickedStroke

from numpy import float_
from numpy.typing import NDArray
import numpy
from skfem import Basis, BilinearForm, DiscreteField, ElementTriP0, ElementTriP1, ElementTriP2, FacetBasis, Functional, InteriorFacetBasis, LinearForm, Mesh, MeshTri, condense, solve
from skfem.helpers import grad
import skfem
import skfem.utils

from pycoimset.util import depends_on, notify_property_update, tracks_dependencies

from .forms import L, a, a_w, e_fac, e_int, ge_int


def refine_mesh(mesh: skfem.MeshTri, wgt: NDArray[float_], tol: float, vol: Optional[NDArray[float_]] = None, max_frac: Optional[float] = None) -> skfem.MeshTri:
    sort_idx = numpy.argsort(wgt) if vol is None else numpy.argsort(wgt / vol)
    sorted_wgt = wgt[sort_idx]
    split_idx = numpy.searchsorted(sorted_wgt, tol / len(sorted_wgt), side='right')
    if max_frac is not None:
        split_idx = max(int(split_idx), math.floor((1 - max_frac) * len(sorted_wgt)))
    where = sort_idx[split_idx:]
    return cast(skfem.MeshTri, mesh.refined(where))


class FunctionSpaces:
    p0: Basis
    p1: Basis
    p2: Basis

    def __init__(self, mesh: MeshTri):
        self.p0 = Basis(mesh, ElementTriP0())
        self.p1 = Basis(mesh, ElementTriP1())
        self.p2 = Basis(mesh, ElementTriP2())


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

        #: Objective function evaluation statistic.
        obj: EvaluationStatistic = field(default_factory=EvaluationStatistic)

        #: Gradient evaluation statistic.
        grad: EvaluationStatistic = field(default_factory=EvaluationStatistic)

    _mesh: MeshTri
    _spc: FunctionSpaces
    _tol: Tolerances
    _stats: Statistics
    _w: numpy.ndarray
    _f: float

    def __init__(self, mesh: Mesh, ctrl_dof: numpy.ndarray,
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
        assert isinstance(mesh, MeshTri)

        self._mesh = mesh.with_boundaries({
            'dirichlet': lambda x: x[0] <= x[1],
            'neumann': lambda x: x[0] > x[1]
        })
        self._spc = FunctionSpaces(self._mesh)
        self._w = ctrl_dof
        self._f = 0.02
        self._tol = PoissonEvaluator.Tolerances(obj_tol, grad_tol)
        self._stats = PoissonEvaluator.Statistics()

    @property
    def stats(self) -> 'PoissonEvaluator.Statistics':
        '''Execution statistics.'''
        return self._stats

    @property
    def mesh(self) -> MeshTri:
        '''Control mesh.'''
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: MeshTri) -> None:
        # Mark relevant boundaries.
        mesh = mesh.with_boundaries({
            'dirichlet': lambda x: x[0] <= x[1],
            'neumann': lambda x: x[0] > x[1]
        })

        # Set up new function spaces.
        s = FunctionSpaces(mesh)

        # Interpolate control to new mesh.
        cell_ctrl_dofs = self._w[self._spc.p0.element_dofs].squeeze(0)
        x_center = mesh.mapping().F(numpy.array([[1/3], [1/3]])).squeeze(-1)
        finder = self._mesh.element_finder()
        idx_parent = numpy.concatenate([
            finder(*x) for x in numpy.array_split(
                x_center, math.ceil(x_center.shape[1] / 100), axis=1
            )
        ])
        self._w = numpy.empty(s.p0.N, dtype=float)
        self._w[s.p0.element_dofs] = cell_ctrl_dofs[idx_parent]
        del cell_ctrl_dofs, x_center, finder, idx_parent

        # Update function spaces
        self._spc = s

        # Replace old mesh.
        self._mesh = mesh

        # Reset everything.
        notify_property_update(self, 'mesh')

    @depends_on(mesh)
    @cached_property
    def vol(self) -> numpy.ndarray:
        '''Cell volumes.'''
        @Functional
        def vol(_):
            return 1.0
        return vol.elemental(self._spc.p0)

    @property
    def spaces(self) -> FunctionSpaces:
        '''Function spaces.'''
        return self._spc

    @property
    def ctrl(self) -> numpy.ndarray:
        '''DOF vector of the control function.'''
        return self._w

    @property
    def tol(self) -> 'PoissonEvaluator.Tolerances':
        '''Evaluation tolerances.'''
        return self._tol

    @tol.setter
    def tol(self, tol: 'PoissonEvaluator.Tolerances') -> None:
        self._tol = tol

    @depends_on(mesh)
    @cached_property
    def pdesol(self) -> numpy.ndarray:
        '''DOF vector of the PDE solution.'''
        time_start = time.perf_counter()
        A = BilinearForm(lambda u, v, w: a(u, v, w.w)).assemble(
            self._spc.p1,
            w=self._spc.p1.with_element(self._spc.p0.elem()
                                        ).interpolate(self._w)
        )
        b = LinearForm(lambda v, w: L(v, w.f)).assemble(
            self._spc.p1,
            f=self._f
        )
        sys = condense(A, b, D=self._spc.p1.get_dofs('dirichlet'))
        solver = skfem.utils.solver_iter_pcg(atol=0, tol=1e-3)
        result = cast(numpy.ndarray, solve(*sys, solver=solver))    # type: ignore
        time_end = time.perf_counter()
        self._stats.pdesol.time += time_end - time_start
        self._stats.pdesol.num += 1
        return result

    @depends_on(mesh)
    @cached_property
    def qpdesol(self) -> numpy.ndarray:
        '''DOF vector of the higher order PDE solution.'''
        time_start = time.perf_counter()
        A = BilinearForm(lambda u, v, w: a(u, v, w.w)).assemble(
            self._spc.p2,
            w=self._spc.p2.with_element(self._spc.p0.elem()
                                        ).interpolate(self._w)
        )
        b = LinearForm(lambda v, w: L(v, w.f)).assemble(
            self._spc.p2,
            f=self._f
        )
        sys = condense(A, b, D=self._spc.p2.get_dofs('dirichlet'))
        solver = skfem.utils.solver_iter_pcg(atol=0, tol=1e-3)
        result = cast(numpy.ndarray, solve(*sys, solver=solver))    # type: ignore
        time_end = time.perf_counter()
        self._stats.qpdesol.time += time_end - time_start
        self._stats.qpdesol.num += 1
        return result

    @depends_on(pdesol)
    @cached_property
    def obj(self) -> float:
        '''Objective function value.'''
        time_start = time.perf_counter()
        result = Functional(lambda w: L(w.y, self._f)).assemble(
            self._spc.p1,
            y=self._spc.p1.interpolate(self.pdesol)
        )
        time_end = time.perf_counter()
        self._stats.obj.time += time_end - time_start
        self._stats.obj.num += 1
        return result

    @property
    def adjsol(self) -> numpy.ndarray:
        '''
        Adjoint solution DOF vector.

        Remarks
        -------
        Because of the compliance objective, the adjoint solution is
        always equal to the PDE solution.
        '''
        return self.pdesol

    @property
    def qadjsol(self) -> numpy.ndarray:
        '''Higher-order adjoint solution DOF vector.'''
        return self.qpdesol

    @depends_on(pdesol, vol)
    @cached_property
    def grad(self) -> numpy.ndarray:
        '''Gradient vector (with respect to DOFs).'''
        pdesol = self.pdesol
        adjsol = self.adjsol
        time_start = time.perf_counter()
        B = -BilinearForm(lambda u, v, w: a_w(w.y, u, v)).assemble(
            self._spc.p1,
            self._spc.p0,
            y=self._spc.p1.interpolate(pdesol)
        )
        result = B.dot(adjsol)
        time_end = time.perf_counter()
        self._stats.grad.time += time_end - time_start
        self._stats.grad.num += 1
        return result

    @depends_on(mesh, qpdesol, pdesol)
    @cached_property
    def objerr(self) -> numpy.ndarray:
        '''Cellwise representation of objective error.'''
        # Retrieve quadrature bases.
        s = self._spc
        m = self._mesh

        # Project higher-order adjoint solution onto P1.
        z = self.qadjsol
        zh = cast(numpy.ndarray, s.p1.project(
            s.p1.with_element(s.p2.elem()).interpolate(z)
        ))

        # Interior residual.
        eta_int = Functional(lambda w: e_int(w.z, w.zh, w.f)).elemental(
            s.p2,
            f=self._f,
            z=s.p2.interpolate(z),
            zh=s.p2.with_element(s.p1.elem()).interpolate(zh)
        )

        # Set up buffer for facet terms.
        eta_fac = numpy.zeros(m.nfacets)

        # Set up functional for facet terms.
        facet_func = Functional(lambda w: e_fac(
            w.y, w.z, w.zh, w.w, (-1)**(w.idx) * w.n
        ))

        for basis_type, subset, side in (
            (InteriorFacetBasis, None, 0),
            (InteriorFacetBasis, None, 1),
            (FacetBasis, 'dirichlet', 0)
        ):
            p2 = basis_type(
                self._mesh, s.p2.elem(), side=side, facets=subset
            )
            p1 = basis_type(
                self._mesh, s.p1.elem(), side=side, facets=subset,
                quadrature=p2.quadrature
            )
            p0 = basis_type(
                self._mesh, s.p0.elem(), side=side, facets=subset,
                quadrature=p2.quadrature
            )

            eta_fac[p2.find] = facet_func.elemental(
                p2,
                y=p1.interpolate(self.pdesol),
                z=p2.interpolate(z),
                zh=p1.interpolate(zh),
                w=p0.interpolate(self._w),
                idx=side
            )

        # Map per-facet quantities to their incident cells.
        eta_bnd = numpy.sum(eta_fac[m.t2f], axis=0) / 2

        return eta_int - eta_bnd

    @depends_on(mesh, pdesol, qpdesol)
    @cached_property
    def graderr(self) -> numpy.ndarray:
        '''Gradient L^1 error estimator.'''
        # Retrieve spaces and mesh.
        s = self.spaces
        m = self.mesh

        # Get DOF vectors.
        y = self.qpdesol
        yh = self.pdesol
        z = self.qadjsol
        zh = self.adjsol

        # Set up quadrature spaces for interior terms.
        q2 = s.p2
        q1 = s.p2.with_element(s.p1.elem())

        # Assemble interior residual terms.
        int_func = Functional(lambda w: ge_int(w.y, w.yh, w.z, w.zh))
        eta_int = int_func.elemental(
            q2,
            y=q2.interpolate(y),
            yh=q1.interpolate(yh),
            z=q2.interpolate(z),
            zh=q1.interpolate(zh)
        )

        return numpy.abs(eta_int)

    def eval_obj(self) -> float:
        '''
        Evaluate objective to given tolerance.
        '''
        while ((err := abs(numpy.sum((eta := self.objerr)).item()))
               > self._tol.obj):
            self.mesh = refine_mesh(self.mesh, numpy.abs(eta), self._tol.obj, 0.01)
        return err

    def eval_grad(self) -> float:
        '''
        Evaluate objective to given tolerance.
        '''
        while ((err := numpy.sum((eta := self.graderr)).item())
               > self._tol.grad):
            print(err)
            self.mesh = refine_mesh(self.mesh, numpy.abs(eta), self._tol.grad, 0.01)
        return err
