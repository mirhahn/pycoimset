# PyCoimset Example "Poisson Design": PDE-related functionality
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
Basic forms for the FEM solver.
'''

from typing import cast

import numpy
from skfem.helpers import dot, grad


__all__ = [
    'a', 'a_w', 'L', 'e_int', 'e_fac', 'ge_int'
]


eps = 0.1


def k(w):
    '''
    Diffusivity.

    Parameters
    ----------
    w : array-like
        Control value. Should be between 0 and 1.
    '''
    return eps + (1 - eps) * w


def dkdw():
    '''
    Derivative of diffusivity with respect to control.
    '''
    return (1 - eps)


def a(u, v, w):
    '''
    Bilinear left-hand side form.

    Parameters
    ----------
    u
        PDE solution.
    v
        Test function.
    w
        Control function.
    '''
    return dot(k(w) * grad(u), grad(v))     # type: ignore


def a_w(u, v, dw=1.0):
    '''
    Derivative of bilinear form with respect to control.

    Parameters
    ----------
    u
        PDE solution.
    v
        Test function.
    dw
        Control perturbation. Defaults to `1.0`.
    '''
    return dot(dkdw() * dw * grad(u), grad(v))  # type:ignore


def L(v, f):
    '''
    Linear right-hand side form.

    Parameters
    ----------
    v
        Test function.
    f
        Source density.
    '''
    return f * v


def e_int(z, zh, f):
    '''
    Interior term of DWR error estimator for main system.

    Parameters
    ----------
    z
        Higher order adjoint solution.
    zh
        Projection of `z` to P1.
    f
        Source density.
    '''
    return f * (z - zh)


def e_fac(y, z, zh, w, n):
    '''
    Facet term of DWR error estimator for main system.

    Parameters
    ----------
    y
        P1 approximation of PDE solution.
    z
        Higher order adjoint solution.
    zh
        Projection of `z` to P1.
    w
        Control function.
    n
        Outer unit normal.
    '''
    return dot(k(w) * grad(y), n) * (z - zh)


def ge_int(y, yh, z, zh):
    '''
    Interior term of gradient error estimator.

    Parameters
    ----------
    zh
        P1 approximation of averaged solution.
    y
        P2 approximation of difference solution
    yh
        P1 approximation of difference solution

    Remarks
    -------
    This term is symmetric in the sense that the "averaged solution"
    can be either the PDE solution or the adjoint state.
    '''
    gy, gyh, gz, gzh = (cast(numpy.ndarray, grad(f)) for f in (y, yh, z, zh))
    dgy = gy - gyh
    dgz = gz - gzh
    return dkdw() * (dot(dgy, gzh)
                     + dot(dgz, gyh))
#                     + dot(dgy, dgz))
