'''Measure space implementation for the Poisson problem.'''

from functools import cached_property, singledispatchmethod
import operator
from types import NotImplementedType
from typing import Generic, Optional, Protocol, Self, TypeVar, cast

import numpy
from numpy.typing import ArrayLike
import pycoimset.typing
import pycoimset.util.weakref as weakref
import skfem


__all__ = [
    'common_mesh',
    'Mesh',
    'SimilaritySpace',
    'SignedMeasure',
    'SimilarityClass',
    'BoolArrayClass'
]


M = TypeVar('M', bound=skfem.MeshTri)


class Mesh(Generic[M]):
    mesh: M
    _parent: Optional[weakref.ref[Self]]
    _children: set[weakref.ref[Self]]
    _elmap: dict[weakref.ref[M], numpy.ndarray]

    def __init__(self, mesh: M, parent: Optional[Self] = None):
        self.mesh = mesh
        self._elmap = dict[weakref.ref[M], numpy.ndarray]()
        self._children = set[weakref.ref[Mesh[M]]]()
        self._parent = None
        self.parent = parent

    @property
    def parent(self) -> Optional[Self]:
        '''Parent of a refined mesh.'''
        if self._parent is None:
            return None
        return self._parent()

    @parent.setter
    def parent(self, parent: Optional[Self]) -> None:
        def clear_parent(_):
            self._parent = None
        if self._parent is not None and (p := self._parent()) is not None:
            p._children.remove(weakref.hashref(self))
        if parent is None:
            self._parent = None
        else:
            parent._children.add(weakref.hashref(self, weakref.weak_key_deleter(parent._children)))
            self._parent = weakref.ref(parent, clear_parent)

    @cached_property
    def p0_basis(self) -> skfem.Basis:
        '''Piecewise constant basis on the given mesh.'''
        return skfem.Basis(self.mesh, skfem.ElementTriP0())

    @cached_property
    def p1_basis(self) -> skfem.Basis:
        '''Piecewise constant basis on the given mesh.'''
        return skfem.Basis(self.mesh, skfem.ElementTriP1())

    @cached_property
    def element_measure(self) -> numpy.ndarray:
        '''Array of element volumes.'''
        return cast(numpy.ndarray, skfem.Functional(lambda _: 1.0).elemental(
            skfem.Basis(self.mesh, skfem.ElementTriP0())
        ))

    @cached_property
    def measure(self) -> float:
        '''Total measure of domain.'''
        return self.element_measure.sum()

    def is_child(self, parent: Self | M) -> bool:
        '''Test whether this mesh is a refinement of another.'''
        if isinstance(parent, Mesh):
            pred = lambda x: x is parent
        else:
            pred = lambda x: x.mesh is parent
        cur = self
        while cur is not None and not pred(cur):
            cur = cur.parent
        return cur is not None

    def element_map(self, child: 'Mesh[M] | M') -> numpy.ndarray:
        '''
        Element map from a refinement of this mesh to this mesh.

        Parameters
        ----------
        child : Mesh[M] | M
            A refinement of this mesh.

        Returns
        -------
            A `numpy.ndarray` with one integer element per child
            element, indicating the index of the parent element
            containing it.

        Remarks
        -------
            The mapping is constructed based on the centroids of the
            mesh.
        '''
        # Normalize argument.
        if isinstance(child, Mesh):
            child_mesh = child.mesh
        else:
            child_mesh = child
        cref = weakref.hashref(
            child_mesh, weakref.weak_key_deleter(self._elmap)
        )

        # Try using a cached element map.
        if child in self._elmap:
            return self._elmap[child]

        # Construct a list of midpoints.
        refdom = child_mesh.init_refdom()
        centroids = child_mesh.mapping().F(
            numpy.mean(refdom.p, axis=1, keepdims=True)
        ).squeeze(-1)

        # Find parent elements
        idx = self.mesh.element_finder()(*centroids)

        # Cache result.
        self._elmap[cref] = idx
        return idx


class MeshDependent(Protocol[M]):
    @property
    def mesh(self) -> Mesh[M]:
        '''Mesh on which this class is defined.'''
        ...

    def adapt(self, mesh: Mesh[M]) -> Self:
        '''Adapts the class to a more refined mesh.'''
        ...


def common_mesh(mesh: Mesh[M] | MeshDependent[M],
                *other: Mesh[M] | MeshDependent[M]) -> Mesh[M]:
    '''
    Find most refined mesh among members of a common refinement hierarchy.
    '''
    parent = mesh if isinstance(mesh, Mesh) else mesh.mesh
    for cand in other:
        if not isinstance(cand, Mesh):
            cand = cand.mesh
        if not cand.is_child(parent):
            parent = cand
    return parent


class SimilaritySpace(pycoimset.typing.SimilaritySpace):
    '''
    Similarity space for the Poisson problem.
    '''
    _mesh: Mesh

    def __init__(self, mesh: skfem.MeshTri):
        self._mesh = Mesh(mesh)

    @property
    def mesh(self) -> Mesh:
        '''Current main control mesh.'''
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh
        delattr(self, 'empty_class')
        delattr(self, 'universal_class')

    @property
    def measure(self) -> float:
        '''Measure of the universal similarity class.'''
        return self.mesh.measure

    @cached_property
    def empty_class(self) -> pycoimset.typing.SimilarityClass[Self]:
        '''Empty similarity class.'''
        return BoolArrayClass(self, self._mesh, False)

    @cached_property
    def universal_class(self) -> pycoimset.typing.SimilarityClass[Self]:
        '''Universal similarity class.'''
        return BoolArrayClass(self, self._mesh, True)


class SimilarityClass(pycoimset.typing.SimilarityClass[SimilaritySpace],
                      MeshDependent, Protocol):
    pass


class BoolArrayClass(SimilarityClass):
    '''
    Encodes a similarity class with an array of booleans.
    '''
    _space: SimilaritySpace
    _mesh: Mesh
    _flag: numpy.ndarray
    _ref: dict[weakref.ref[Mesh], Self]

    def __init__(self, space: SimilaritySpace, mesh: Optional[Mesh] = None,
                 flag: ArrayLike = False):
        if mesh is None:
            mesh = space.mesh
        self._space = space
        self._mesh = mesh
        self._flag = numpy.broadcast_to(
            numpy.asarray(flag, dtype=bool),
            mesh.mesh.nelements
        )
        self._ref = dict[weakref.ref[Mesh], Self]()

    def adapt(self, mesh: Mesh) -> Self:
        '''Adapt to a refined mesh.'''
        if mesh is self.mesh:
            return self

        try:
            return self._ref[weakref.hashref(mesh)]
        except KeyError:
            element_map = self.mesh.element_map(mesh)
            buf = self.flag[element_map]
            retval = BoolArrayClass(self.space, mesh, buf)
            self._ref[
                weakref.hashref(mesh, weakref.weak_key_deleter(self._ref))
            ] = retval
            return retval

    @property
    def space(self) -> SimilaritySpace:
        '''Underlying similarity space.'''
        return self._space

    @property
    def mesh(self) -> Mesh:
        '''Definition mesh.'''
        return self._mesh

    @property
    def flag(self) -> numpy.ndarray:
        '''Boolean array indicating set membership for each mesh element.'''
        return self._flag

    @cached_property
    def measure(self) -> float:
        return self.mesh.element_measure[self.flag].sum()

    def _subset_nohint(self, meas_low: float, meas_high: float
                       ) -> 'BoolArrayClass':
        '''
        Choose subset within a given size range without hint.

        Parameters
        ----------
        meas_low : float
            Lower bound on subset measure. Must be strictly less than
            `meas_high` and must not exceed measure of this class.
        meas_high : float
            Upper bound on subset measure. Must be strictly greater
            than `meas_low`.

        Remarks
        -------
            This method may internally refine the mesh.
        '''
        assert meas_low < meas_high

        # Set up variable for refined set.
        refined = self

        while True:
            # Find element indices and sort by element measure.
            tind = numpy.flatnonzero(refined.flag)
            meas = refined.mesh.element_measure[tind]

            sidx = numpy.argsort(-meas)
            tind = tind[sidx]
            meas = meas[sidx]
            del sidx

            # Search for break point in cumulative measure array.
            cum_meas = numpy.cumsum(meas)
            end = int(numpy.searchsorted(cum_meas, meas_high, side='right'))
            cur_meas = 0.0 if end == 0 else cum_meas[end - 1]

            if cur_meas >= meas_low:
                break

            # Refine mesh
            end = min(end + 1, numpy.searchsorted(cum_meas,
                                                  0.3 * cum_meas[-1],
                                                  side='right'))
            base_mesh = refined.mesh.mesh.refined(tind[:end])
            mesh = Mesh(base_mesh, parent=self.mesh)
            refined = self.adapt(mesh)

        # Return result.
        flag = numpy.zeros_like(refined.flag)
        flag[tind[:end]] = True
        return BoolArrayClass(self.space, refined.mesh, flag)

    def _subset_hint(self, meas_low: float, meas_high: float,
                     hint: 'SignedMeasure') -> 'BoolArrayClass':
        '''
        Choose subset within a given size range without hint.

        Parameters
        ----------
        meas_low : float
            Lower bound on subset measure. Must be strictly less than
            `meas_high` and must not exceed measure of this class.
        meas_high : float
            Upper bound on subset measure. Must be strictly greater
            than `meas_low`.
        hint : SignedMeasure
            Refined gradient measure to be used as a hint for where to
            best refine.

        Remarks
        -------
            This method may internally refine the mesh.
        '''
        assert meas_low < meas_high

        # Establish mapping to hint mesh.
        emap = self.mesh.element_map(hint.mesh)

        # Establish average gradient value.
        gval = numpy.zeros(self.mesh.mesh.nelements)
        numpy.add.at(gval, emap, hint.elemental_integrals)
        gval /= self.mesh.element_measure

        # Establish number of child elements.
        cnum = numpy.bincount(emap, minlength=self.mesh.mesh.nelements)

        # Set up variable for refined set.
        refined = self

        while True:
            # Find element indices and sort by element measure.
            tind = numpy.flatnonzero(refined.flag)
            tval = gval[tind]
            meas = self.mesh.element_measure[tind]

            sidx = numpy.argsort(tval)
            tind = tind[sidx]
            tval = tval[sidx]
            meas = meas[sidx]
            del sidx

            # Search for break point in cumulative measure array.
            cum_meas = numpy.cumsum(meas)
            end = int(numpy.searchsorted(cum_meas, meas_high, side='right'))
            cur_meas = 0.0 if end == 0 else cum_meas[end - 1]

            if cur_meas >= meas_low:
                break

            # Refine mesh
            sidx = numpy.argsort(-meas)
            tind = tind[sidx]
            meas = meas[sidx]
            end = numpy.searchsorted(cum_meas, 0.3 * cum_meas[-1],
                                     side='right')

            mesh = Mesh(refined.mesh.mesh.refined(tind[:end]),
                        parent=self.mesh)
            rmap = refined.mesh.element_map(mesh)
            rnum = numpy.bincount(rmap, minlength=refined.mesh.mesh.nelements)
            gval = gval[rmap]
            cnum = (cnum / rnum)[rmap]
            refined = self.adapt(mesh)

        # Return result.
        flag = numpy.zeros_like(refined.flag)
        flag[tind[:end]] = True
        return BoolArrayClass(self.space, refined.mesh, flag)

    def subset(self, meas_low: float, meas_high: float,
               hint: Optional[pycoimset.SignedMeasure[SimilaritySpace]] = None
               ) -> 'BoolArrayClass':
        '''Choose subset within a given size range.'''
        # Catch type errors for the hint.
        if hint is not None and (not isinstance(hint, SignedMeasure)
                                 or hint.space is not self.space):
            hint = None

        # Call subroutine based on availability of hint.
        if hint is None:
            return self._subset_nohint(meas_low, meas_high)
        return self._subset_hint(meas_low, meas_high, hint)

    def __invert__(self) -> Self:
        '''Complement of the similarity class.'''
        return BoolArrayClass(self.space, self.mesh, ~self.flag)

    def _binop(self, op, other) -> 'BoolArrayClass | NotImplementedType':
        '''Perform arbitrary binary logical operation.'''
        if not (isinstance(other, BoolArrayClass)
                and other.space is self.space):
            return NotImplemented

        mesh = common_mesh(self.mesh, other.mesh)
        self = self.adapt(mesh)
        other = other.adapt(mesh)

        return BoolArrayClass(self.space, mesh, op(self.flag, other.flag))

    def __or__(self, other) -> 'BoolArrayClass | NotImplementedType':
        '''Union.'''
        return self._binop(operator.or_, other)

    def __and__(self, other) -> 'BoolArrayClass | NotImplementedType':
        '''Union.'''
        return self._binop(operator.and_, other)

    def __xor__(self, other) -> 'BoolArrayClass | NotImplementedType':
        '''Union.'''
        return self._binop(operator.xor, other)

    def __sub__(self, other) -> 'BoolArrayClass | NotImplementedType':
        '''Union.'''
        return self._binop(lambda x, y: x & ~y, other)

    def __rsub__(self, other) -> 'BoolArrayClass | NotImplementedType':
        '''Union.'''
        return self._binop(lambda x, y: y & ~x, other)


class SignedMeasure(pycoimset.typing.SignedMeasure, MeshDependent):
    '''
    Encodes a signed measure as a piecewise constant function.
    '''
    _space: SimilaritySpace
    _mesh: Mesh
    _dof: numpy.ndarray
    _ref: dict[weakref.ref[Mesh], Self]

    def __init__(self, space: SimilaritySpace, mesh: Mesh,
                 value: ArrayLike = 0.0):
        self._space = space
        self._mesh = mesh
        self._dof = numpy.broadcast_to(value, self.mesh.p0_basis.N)
        self._ref = dict[weakref.ref[Mesh], Self]()

    @property
    def space(self) -> SimilaritySpace:
        '''Underlying similarity space.'''
        return self._space

    @property
    def mesh(self) -> Mesh:
        '''Definition mesh.'''
        return self._mesh

    @property
    def dof(self) -> numpy.ndarray:
        '''Degrees of freedom of the density function.'''
        return self._dof

    def adapt(self, mesh: Mesh) -> 'SignedMeasure':
        '''
        Project this measure to a finer mesh.

        Parameters
        ----------
        mesh : Mesh
            The refined mesh to project to.
        '''
        if mesh is self.mesh:
            return self

        try:
            return self._ref[weakref.hashref(mesh)]
        except KeyError:
            element_map = self.mesh.element_map(mesh)
            buf = numpy.empty(mesh.p0_basis.N)
            old_idx = self.mesh.p0_basis.element_dofs[0]
            new_idx = mesh.p0_basis.element_dofs[0]
            buf[new_idx] = self.dof[old_idx][element_map]
            retval = SignedMeasure(self.space, mesh, buf)
            self._ref[
                weakref.hashref(mesh, weakref.weak_key_deleter(self._ref))
            ] = retval
            return retval

    @cached_property
    def elemental_integrals(self) -> numpy.ndarray:
        '''Elemental integrals of the density function.'''
        p0 = self.mesh.p0_basis
        return skfem.Functional(lambda w: w.g).elemental(
            p0, g=p0.interpolate(self.dof)
        )

    @singledispatchmethod
    def __call__(self, simcls: SimilarityClass) -> float:
        '''Return measure of a similarity class.'''
        raise NotImplementedError(f'Cannot measure {type(simcls).__name__}')

    @__call__.register
    def _(self, simcls: BoolArrayClass) -> float:
        # Ensure that both objects share a similarity space.
        if self.space is not simcls.space:
            raise NotImplementedError('Similarity class and signed measure '
                                      'must refere to the same space.')

        # Adapt to common mesh.
        mesh = common_mesh(self.mesh, simcls.mesh)
        self = self.adapt(mesh)
        simcls = simcls.adapt(mesh)

        # Calculate integral by summing elemental integrals.
        return self.elemental_integrals[simcls.flag].sum()

    def _cmp(self, op, other: float | Self) -> BoolArrayClass:
        '''Perform generic comparison operation to obtain similarity class.'''
        # NOTE: Presumes that DOFs of P0 are elemental function values
        # in element index order.
        if isinstance(other, SignedMeasure):
            if self.space is not other.space:
                raise NotImplementedError('Cannot compare measures on '
                                          'different spaces.')
            mesh = common_mesh(self.mesh, other.mesh)
            self = self.adapt(mesh)
            other_dof = other.adapt(mesh).dof
        else:
            other_dof = other
        return BoolArrayClass(self.space, self.mesh, op(self.dof, other_dof))

    def _arith(self, op, *args: float | Self) -> Self:
        '''Perform generic arithmetic operation to obtain signed measure.'''
        # NOTE: Presumes that DOFs of P0 are elemental function values
        # in case of nonlinear operations.

        # Find common mesh:
        meshes = [self.mesh]
        for arg in args:
            if not isinstance(arg, SignedMeasure):
                continue
            if arg.space is not self.space:
                raise NotImplementedError('Cannot perform arithmetic between '
                                          'measures on different spaces.')
            meshes.append(arg.mesh)
        mesh = common_mesh(*meshes)

        # Generate list of DOF arrays.
        arg_dofs = [
            self.adapt(mesh).dof,
            *(
                arg.adapt(mesh).dof if isinstance(arg, SignedMeasure)
                else arg for arg in args
            )
        ]

        # Execute operation and create new SignedMeasure.
        return SignedMeasure(self.space, mesh, op(*arg_dofs))

    def __lt__(self, other) -> BoolArrayClass | NotImplementedType:
        '''Strict sublevel set.'''
        return self._cmp(operator.lt, other)

    def __le__(self, other) -> BoolArrayClass | NotImplementedType:
        '''Sublevel set.'''
        return self._cmp(operator.le, other)

    def __gt__(self, other) -> BoolArrayClass | NotImplementedType:
        '''Strict superlevel set.'''
        return self._cmp(operator.gt, other)

    def __ge__(self, other) -> BoolArrayClass | NotImplementedType:
        '''Superlevel set.'''
        return self._cmp(operator.ge, other)

    def __add__(self, other) -> Self | NotImplementedType:
        '''Add measures or shift density function.'''
        if not isinstance(other, (float, int, SignedMeasure)):
            return NotImplemented
        return self._arith(operator.add, other)

    def __sub__(self, other) -> Self | NotImplementedType:
        '''Subtract measures or shift density function.'''
        if not isinstance(other, (float, int, SignedMeasure)):
            return NotImplemented
        return self._arith(operator.sub, other)

    def __rsub__(self, other) -> Self | NotImplementedType:
        '''Subtract measures or shift density function.'''
        if not isinstance(other, (float, int, SignedMeasure)):
            return NotImplemented
        return self._arith(lambda x, y: y - x, other)

    def __mul__(self, other) -> Self | NotImplementedType:
        '''Scale measure or multiply density functions.'''
        if not isinstance(other, (float, int, SignedMeasure)):
            return NotImplemented
        return self._arith(operator.mul, other)

    def __truediv__(self, other) -> Self | NotImplementedType:
        '''Scale measure or multiply density functions.'''
        if not isinstance(other, (float, int, SignedMeasure)):
            return NotImplemented
        return self._arith(operator.truediv, other)
