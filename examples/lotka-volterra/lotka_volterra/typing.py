# PyCoimset Example "Lotka-Volterra": Problem-specific code
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
Typing helpers for objects that behave like SciPy OdeSolution.
'''

from typing import Protocol

from numpy.typing import ArrayLike, NDArray


class OdeSolutionLike(Protocol):
    '''
    Behaves like `scipy.integrate.OdeSolution` with `deriv` extension.
    '''

    #: Time instants between which local interpolants are defined.
    #: Must be strictly increasing.
    ts: NDArray

    @property
    def t_min(self) -> float:
        '''Start of interpolation time range.'''
        return self.ts[0]

    @property
    def t_max(self) -> float:
        '''End of interpolation time range.'''
        return self.ts[-1]

    def __call__(self, t: ArrayLike) -> NDArray:
        '''Evaluate the solution.'''
        ...

    def deriv(self, t: ArrayLike) -> NDArray:
        '''Evaluate the first derivative of the solution.'''
        ...
