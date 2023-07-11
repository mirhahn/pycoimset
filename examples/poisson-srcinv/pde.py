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
import time
from typing import NamedTuple, Optional, cast

import numpy
import scipy.sparse
import skfem
from skfem.helpers import dot, grad


def poisson_lhs(y, v, k):
    '''
    Left hand side of the Poisson equation.

    Parameters
    ----------
    y
        Trial/Solution function.
    v
        Test function.
    k
        Diffusivity coefficient.

    Remarks
    -------
    This form is linear in `y` and `v` and can therefore also be used
    to generate adjoints and derivatives. It is affine in `k`.
    '''
    # Interpolate control function.
    return dot((0.999 * k + 0.001) * grad(y), grad(v))


def poisson_lhs_deriv_diffusivity(y, v, dk):
    '''Derivative of `poisson_rhs` with respect to `k`.'''
    return dot(0.999 * dk * grad(y), grad(v))


poisson_lhs_deriv_trial = poisson_lhs


poisson_lhs_deriv_test = poisson_lhs


def poisson_rhs(v, f):
    '''
    Right hand side of the Poisson equation.

    Parameters
    ----------
    v
        Test function.
    f
        Source density.
    '''
    return f * v


def objective(y, f):
    '''
    Objective functional.

    Parameters
    ----------
    y
        Poisson equation solution.
    f
        Source density.
    '''
    return y * f


objective_deriv_solution = objective


class PoissonFormsTuple(NamedTuple):
    '''Integral forms for the problem.'''
    primal_bilin: skfem.BilinearForm
    primal_lin: skfem.LinearForm
    obj: skfem.Functional
    obj_grad: skfem.LinearForm
    adjoint_bilin_grad2test: skfem.BilinearForm
    adjoint_bilin_test2ctrl: skfem.BilinearForm


PoissonForms = PoissonFormsTuple(
    primal_bilin=skfem.BilinearForm(
        lambda u, v, w: poisson_lhs(u, v, w.k)
    ),
    primal_lin=skfem.LinearForm(lambda v, w: poisson_rhs(v, w.f)),
    obj=skfem.Functional(lambda w: objective(w.y, w.f)),
    obj_grad=skfem.LinearForm(lambda v, w: objective(v, w.f)),
    adjoint_bilin_grad2test=skfem.BilinearForm(
        lambda u, v, w: poisson_lhs_deriv_trial(v, u, w.k)
    ),
    adjoint_bilin_test2ctrl=skfem.BilinearForm(
        lambda u, v, w: poisson_lhs_deriv_diffusivity(w.y, u, v)
    )
)


@dataclass
class FunctionSpaces:
    #: Control function space.
    ctrl: skfem.Basis

    #: PDE solution space.
    pde_sol: skfem.Basis

    #: Quadrature space. This is a parent space which is suitable to
    #: exactly represent all composite functions used in forms.
    quad: skfem.Basis

    @classmethod
    def from_mesh(cls, mesh: skfem.Mesh) -> 'FunctionSpaces':
        '''
        Create default function spaces for a given mesh.

        Parameters
        ----------
        mesh : skfem.Mesh
            Underlying mesh.

        Remarks
        -------
        At the moment, this method is only implemented for triangular
        meshes and will raise `NotImplementedError` for all others.
        '''
        if not isinstance(mesh, skfem.MeshTri):
            raise NotImplementedError()
        return FunctionSpaces(
            ctrl=skfem.Basis(mesh, skfem.ElementTriP0()),
            pde_sol=skfem.Basis(mesh, skfem.ElementTriP1()),
            quad=skfem.Basis(mesh, skfem.ElementTriP2())
        )

    def make_quadrature_spaces(self, quad: Optional[skfem.Basis] = None
                               ) -> 'FunctionSpaces':
        '''
        Create corresponding quadrature spaces.

        This will generate a set of similar spaces with the same
        elements but with the quadrature points of the quadrature
        space. This is useful for form assembly because it guarantees
        that the integral forms are assembled exactly rather than using
        lower-order approximations.

        Parameters
        ----------
        quad : skfem.Basis, optional
            New quadrature space.
        '''
        if quad is None:
            quad = self.quad
        else:
            assert quad.mesh is self.quad.mesh
        return FunctionSpaces(
            ctrl=self.quad.with_element(self.ctrl.elem()),
            pde_sol=self.quad.with_element(self.pde_sol.elem()),
            quad=quad
        )


class AssembledForms:
    _spc: FunctionSpaces
    _qspc: FunctionSpaces

    _k: Optional[numpy.ndarray]
    _f: Optional[float]
    _y: Optional[numpy.ndarray]

    def __init__(self, spaces: FunctionSpaces,
                 quad_spc: Optional[skfem.Basis] = None):
        assert spaces.ctrl.mesh is spaces.pde_sol.mesh
        assert spaces.ctrl.mesh is spaces.quad.mesh

        self._spc = spaces
        self._qspc = spaces.make_quadrature_spaces(quad_spc)

        self._k = None
        self._f = None
        self._y = None

    def _reset(self, *attrs: str) -> None:
        '''Reset properties by name.'''
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @property
    def spaces(self) -> FunctionSpaces:
        '''Basic function spaces.'''
        return self._spc

    @property
    def qspaces(self) -> FunctionSpaces:
        '''Higher-order function spaces.'''
        return self._qspc

    @property
    def k(self) -> Optional[numpy.ndarray]:
        '''Diffusivity parameter.'''
        return self._k

    @k.setter
    def k(self, val: Optional[numpy.ndarray]):
        self._k = val
        self._reset('forward_system',
                    'adjoint_system',
                    'quad_adjoint_system')

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
        self._reset('forward_system',
                    'objective_value',
                    'gradient_vector',
                    'quad_gradient_vector',
                    'adjoint_system',
                    'quad_adjoint_system')

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
        self._reset('objective_value',
                    'adjoint_system',
                    'quad_adjoint_system',
                    'gradient_matrix')

    def get_y(self) -> numpy.ndarray:
        '''Get current solution or raise exception.'''
        assert self._y is not None
        return self._y

    @cached_property
    def forward_system(self) -> skfem.CondensedSystem:
        '''Assembled Poisson system.'''
        # Get bases.
        basis_ctrl = self._spc.ctrl
        basis_sol = self._spc.pde_sol

        # Assemble stiffness matrix and weight vector.
        a_mat = PoissonForms.primal_bilin.assemble(basis_sol, k=basis_ctrl.interpolate(self.get_k()))
        b_vec = PoissonForms.primal_lin.assemble(basis_sol, f=self.get_f())

        # Condense system.
        return skfem.condense(a_mat, b_vec, D=basis_sol.get_dofs(['left', 'top']))

    @cached_property
    def objective_value(self) -> float:
        '''Objective function value.'''
        basis_sol = self._spc.pde_sol
        return PoissonForms.obj.assemble(
            basis_sol,
            f=self.get_f(),
            y=basis_sol.interpolate(self.get_y())
        )

    @cached_property
    def gradient_vector(self) -> numpy.ndarray:
        '''Assembled gradient vector.'''
        return PoissonForms.obj_grad.assemble(self._spc.pde_sol, f=self.get_f())

    @cached_property
    def quad_gradient_vector(self) -> numpy.ndarray:
        '''Assembled higher-order gradient vector.'''
        return PoissonForms.obj_grad.assemble(self._spc.quad, f=self.get_f())

    @cached_property
    def adjoint_system(self) -> skfem.CondensedSystem:
        '''Assembled adjoint system.'''
        a_mat = PoissonForms.adjoint_bilin_grad2test.assemble(
            self._qspc.pde_sol, self._qspc.pde_sol,
            k=self._qspc.ctrl.interpolate(self.get_k())
        )
        b_vec = self.gradient_vector
        return skfem.condense(
            a_mat, b_vec, D=self._qspc.pde_sol.get_dofs(['left', 'top'])
        )

    @cached_property
    def quad_adjoint_system(self) -> skfem.CondensedSystem:
        '''Assembled higher-order adjoint system.'''
        a_mat = PoissonForms.adjoint_bilin_grad2test.assemble(
            self._spc.quad, self._spc.quad,
            k=self._qspc.ctrl.interpolate(self.get_k())
        )
        b_vec = self.quad_gradient_vector
        return skfem.condense(
            a_mat, b_vec, D=self._spc.quad.get_dofs(['left', 'top'])
        )

    @cached_property
    def gradient_matrix(self) -> scipy.sparse.spmatrix:
        '''Assembled adjoint-to-gradient mapping.'''
        return PoissonForms.adjoint_bilin_test2ctrl.assemble(
            self._qspc.pde_sol, self._qspc.ctrl,
            y=self._qspc.pde_sol.interpolate(self.get_y())
        )


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

    def __init__(self, basis: skfem.Basis, dof: numpy.ndarray,
                 obj_tol: float = 1e-5, grad_tol: float = 1e-5):
        '''
        Constructor.

        Parameters
        ----------
        basis : skfem.Basis
            Basis of the control space.
        dof : numpy.ndarray
            Control function DOF vector.
        obj_tol : float, optional
            Tolerance for objective evaluation. Defaults to `1e-5`.
        grad_tol : float, optional
            Tolerance for gradient evaluation. Defaults to `1e-5`.
        '''
        assert isinstance(basis.mesh, skfem.MeshTri)

        self._mesh = basis.mesh
        bctrl = basis
        bsol = skfem.Basis(self._mesh, skfem.ElementTriP1())
        bquad = skfem.Basis(self._mesh, skfem.ElementTriP2())
        spc = FunctionSpaces(
            pde_sol=bsol,
            ctrl=bctrl,
            quad=bquad
        )
        self._forms = AssembledForms(spc)
        self._forms.k = dof
        self._forms.f = 1e-2
        self._tol = PoissonEvaluator.Tolerances(obj_tol, grad_tol)
        self._stats = PoissonEvaluator.Statistics()

    def _propreset(self, *names: str):
        '''Reset named properties by deleting them.'''
        if len(names) == 0:
            names = [key for key, value in type(self).__dict__.items() if isinstance(value, cached_property)]
        for name in names:
            try:
                delattr(self, name)
            except AttributeError:
                pass

    @property
    def stats(self) -> 'PoissonEvaluator.Statistics':
        '''Execution statistics.'''
        return self._stats

    @property
    def mesh(self) -> skfem.Mesh:
        '''Control mesh.'''
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: skfem.Mesh) -> None:
        # Mark relevant boundaries.
        mesh = mesh.with_boundaries({
            'left': lambda x: numpy.isclose(x[0], 0.0),
            'top': lambda x: numpy.isclose(x[1], 1.0)
        })

        # Create a new forms object.
        old_forms = self._forms
        spaces = FunctionSpaces(
            pde_sol=skfem.Basis(mesh, old_forms.spaces.pde_sol.elem()),
            ctrl=skfem.Basis(mesh, old_forms.spaces.ctrl.elem()),
            quad=skfem.Basis(mesh, old_forms.spaces.quad.elem())
        )
        self._forms = AssembledForms(spaces)
        self._forms.f = old_forms.f

        k_interp = old_forms.spaces.ctrl.interpolator(old_forms.k)(spaces.ctrl.global_coordinates())
        self._forms.k = spaces.ctrl.project(k_interp)

        # Replace old mesh.
        self._mesh = mesh

        # Reset everything.
        self._propreset()

    @property
    def space(self) -> FunctionSpaces:
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

    @cached_property
    def pdesol(self) -> numpy.ndarray:
        '''DOF vector of the PDE solution.'''
        if self._forms.k is not self.ctrl:
            self._forms.k = self.ctrl
        time_start = time.perf_counter()
        result = cast(
            numpy.ndarray,
            skfem.solve(*self._forms.forward_system)   # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.pdesol.time += time_end - time_start
        self._stats.pdesol.num += 1
        return result

    @cached_property
    def obj(self) -> float:
        '''Objective function value.'''
        if self._forms.y is not self.pdesol:
            self._forms.y = self.pdesol
        time_start = time.perf_counter()
        result = self._forms.objective_value
        time_end = time.perf_counter()
        self._stats.obj.time += time_end - time_start
        self._stats.obj.num += 1
        return result

    @cached_property
    def adjsol(self) -> numpy.ndarray:
        '''Adjoint solution DOF vector.'''
        if self._forms.k is not self.ctrl:
            self._forms.k = self.ctrl
        if self._forms.y is not self.pdesol:
            self._forms.y = self.pdesol
        time_start = time.perf_counter()
        result = -cast(
            numpy.ndarray,
            skfem.solve(*self._forms.adjoint_system)    # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.adjsol.time += time_end - time_start
        self._stats.adjsol.num += 1
        return result

    @cached_property
    def qadjsol(self) -> numpy.ndarray:
        '''Higher-order adjoint solution DOF vector.'''
        if self._forms.k is not self.ctrl:
            self._forms.k = self.ctrl
        time_start = time.perf_counter()
        result = -cast(
            numpy.ndarray,
            skfem.solve(*self._forms.quad_adjoint_system)   # pyright: ignore
        )
        time_end = time.perf_counter()
        self._stats.qadjsol.time += time_end - time_start
        self._stats.qadjsol.num += 1
        return result

    @cached_property
    def objerr(self) -> numpy.ndarray:
        '''Cellwise representation of objective error.'''
        # Retrieve quadrature bases.
        quad_p0 = self._forms.spaces.ctrl
        quad_p1 = self._forms.spaces.pde_sol
        quad_p2 = self._forms.spaces.quad

        quad_bnd_p1 = [
            skfem.InteriorFacetBasis(quad_p1.mesh, quad_p1.elem(), side=i)
            for i in range(2)
        ]
        quad_bnd_p2 = skfem.InteriorFacetBasis(quad_p2.mesh, quad_p2.elem(),
                                               side=0)

        elem_p1 = self._forms.spaces.pde_sol.elem()
        elem_p2 = quad_p2.elem()

        # Interior residual.
        @skfem.Functional
        def interior_residual(w):
            return w.f * (w.z - w.phi)
        eta_int = interior_residual.elemental(
            self._forms.spaces.quad,
            f=self._forms.f,
            z=self.qadjsol,
            phi=self._forms.qspaces.pde_sol.interpolate(self.adjsol)
        )

        # Edge jump terms.
        @skfem.Functional
        def edge_residual(w):
            h = w.h
            n = w.n
            dw1 = grad(w.u1)
            dw2 = grad(w.u2)
            return (1/2) * dot(n, dw1 - dw2) * (w.z - w.phi)
        eta_bnd = numpy.sqrt(
            edge_residual.elemental(
                quad_bnd_p2,
                u1=quad_bnd_p2.with_element(quad_bnd_p1[0].elem()
                                            ).interpolate(self.pdesol),
                u2=quad_bnd_p2.with_element(quad_bnd_p1[1].elem()
                                            ).interpolate(self.pdesol),
                z=quad_bnd_p2.interpolate(self.qadjsol),
                phi=quad_bnd_p2.with_element(quad_bnd_p1[0].elem()
                                             ).interpolate(self.adjsol)
            )
        )

        # Map per-facet quantities to their incident cells.
        tmp = numpy.zeros(self._mesh.nfacets)
        tmp[quad_bnd_p2.find] = eta_bnd
        eta_bnd = numpy.sum(tmp[self._mesh.t2f], axis=0)

        return numpy.abs(eta_int) + numpy.abs(eta_bnd)

    @cached_property
    def grad(self) -> numpy.ndarray:
        '''Gradient DOF vector.'''
        if self._forms.y is not self.pdesol:
            self._forms.y = self.pdesol
        adj_sol = self.adjsol
        time_start = time.perf_counter()
        result = self._forms.gradient_matrix.dot(adj_sol)
        time_end = time.perf_counter()
        self._stats.grad.time += time_end - time_start
        self._stats.grad.num += 1
        return result

    def eval_obj(self) -> float:
        '''
        Evaluate objective to given tolerance.
        '''
        while (err := numpy.sum((eta := self.objerr))) > self._tol.obj:
            eta_scaled = self._mesh.params()**2 * eta
            sort_idx = numpy.argsort(eta_scaled)
            cum_err = numpy.cumsum(eta_scaled[sort_idx])
            split_idx = numpy.searchsorted(cum_err, 0.7 * cum_err[-1])
            where = sort_idx[split_idx:]

            print(f'Error = {err}; refining {where.size}/{self._mesh.nelements} cells')

            self.mesh = self._mesh.refined(where)
        print(f'Error = {err}')
        return self.obj
