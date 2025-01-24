# PyCoimset Example "Poisson Design": Set functionals
#
# Copyright 2024 Mirko Hahn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Definition of objective and constraint functionals.
'''

from typing import Optional, cast
import numpy

import pycoimset

from pde import PoissonEvaluator
from space import BoolArrayClass, Mesh, SignedMeasure, SimilaritySpace


__all__ = [
    'MeasureFunctional',
    'ObjectiveFunctional'
]


class MeasureFunctional(pycoimset.Functional[SimilaritySpace]):
    _space: SimilaritySpace
    _arg: Optional[BoolArrayClass]
    _val: Optional[float]
    _grad: Optional[SignedMeasure]

    def __init__(self, space: SimilaritySpace):
        self._space = space
        self._arg = None
        self._val = None
        self._grad = None

    @property
    def input_space(self) -> SimilaritySpace:
        '''Similarity space for functional argument.'''
        return self._space

    @property
    def arg(self) -> BoolArrayClass:
        return cast(BoolArrayClass, self._arg)

    @arg.setter
    def arg(self, arg: pycoimset.SimilarityClass[SimilaritySpace] | None):
        if not isinstance(arg, BoolArrayClass):
            raise NotImplementedError('Not implemented for type '
                                      f'{type(arg).__name__}')
        self._arg = arg
        self._val = None
        self._grad = None

    @property
    def val_tol(self) -> float:
        return 0.0

    @val_tol.setter
    def val_tol(self, tol: float):
        pass

    @property
    def grad_tol(self) -> float:
        return 0.0

    @grad_tol.setter
    def grad_tol(self, tol: float):
        pass

    @property
    def grad_tol_type(self) -> pycoimset.ErrorNorm:
        return pycoimset.ErrorNorm.L1

    def get_value(self) -> tuple[float, float]:
        if self._val is None:
            self._val = cast(BoolArrayClass, self._arg).measure
        return self._val, 0.0

    def get_gradient(self) -> tuple[SignedMeasure, float]:
        if self._grad is None:
            arg = cast(BoolArrayClass, self._arg)
            mesh = arg.mesh
            dof = numpy.ones(mesh.p0_basis.N)
            dof[arg.flag] = -1.0
            self._grad = SignedMeasure(arg.space, mesh, dof)
        return self._grad, 0.0


class ObjectiveFunctional(pycoimset.Functional[SimilaritySpace]):
    _space: SimilaritySpace
    _eval: Optional[PoissonEvaluator]
    _arg: Optional[BoolArrayClass]
    _valtol: float
    _gradtol: float

    def __init__(self, space: SimilaritySpace):
        self._space = space
        self._eval = None
        self._arg = None
        self._valtol = numpy.inf
        self._gradtol = numpy.inf

    @property
    def input_space(self) -> SimilaritySpace:
        '''Similarity space for functional argument.'''
        return self._space

    @property
    def arg(self) -> BoolArrayClass:
        if self._arg is None:
            self.arg = BoolArrayClass(self.input_space, self.input_space.mesh)
        return cast(BoolArrayClass, self._arg)

    @arg.setter
    def arg(self, arg: pycoimset.SimilarityClass[SimilaritySpace] | None):
        if not isinstance(arg, BoolArrayClass):
            raise NotImplementedError('Not implemented for type '
                                      f'{type(arg).__name__}')
        self._arg = arg
        self._eval = None

    @property
    def evaluator(self) -> PoissonEvaluator:
        if self._eval is None:
            arg = self.arg
            dof = arg.flag.astype(float)
            self._eval = PoissonEvaluator(arg.mesh.mesh, dof, self._valtol,
                                          self._gradtol)
        return self._eval

    @property
    def val_tol(self) -> float:
        return self._valtol

    @val_tol.setter
    def val_tol(self, tol: float) -> None:
        self._valtol = tol
        if self._eval is not None:
            self._eval.tol = PoissonEvaluator.Tolerances(self._valtol,
                                                         self._gradtol)

    @property
    def grad_tol(self) -> float:
        return self._gradtol

    @grad_tol.setter
    def grad_tol(self, tol: float) -> None:
        self._gradtol = tol
        if self._eval is not None:
            self._eval.tol = PoissonEvaluator.Tolerances(self._valtol,
                                                         self._gradtol)

    @property
    def grad_tol_type(self) -> pycoimset.ErrorNorm:
        return pycoimset.ErrorNorm.L1

    def get_value(self) -> tuple[float, float]:
        self.evaluator.eval_obj()
        return self.evaluator.obj, abs(self.evaluator.objerr.sum())

    def get_gradient(self) -> tuple[SignedMeasure, float]:
        # Evaluate gradient.
        eval = self.evaluator
        eval.eval_grad()

        dof = -(2 * eval.ctrl - 1) * eval.grad / eval.vol
        grad = SignedMeasure(self.input_space,
                             Mesh(eval.mesh, parent=self.arg.mesh),
                             dof)
        return grad, numpy.abs(eval.graderr).sum()
