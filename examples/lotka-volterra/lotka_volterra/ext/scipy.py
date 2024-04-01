# PyCoimset Example "Lotka-Volterra": SciPy extensions
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
Required extensions for SciPy.

The Lotka-Volterra example requires some ODE interpolant functionality that is
not provided by SciPy at the time of writing. Most notably, we require time
derivatives of the interpolants for error estimation.

This module is designed to inject the necessary code into the SciPy classes at
runtime. This is a very fragile procedure that can be broken by any change to
SciPy's internal API.
'''

import itertools
import functools
from typing import Callable

import numpy
import scipy.integrate


def deriv_impl_rkdenseoutput(self, t: numpy.ndarray):
    '''
    Forward Jacobian implementation for SciPy's RkDenseOutput.
    '''
    x = (t - self.t_old) / self.h
    if t.ndim == 0:
        p = numpy.empty((self.order + 1,))
        p[0] = 1.0
        p[1:] = x
        p = numpy.cumprod(p) * numpy.arange(1, self.order + 2)
    else:
        p = numpy.empty((self.order + 1, t.size))
        p[0, :] = 1.0
        p[1:, :] = x.reshape((1, -1))
        p = numpy.cumprod(p, axis=0) \
            * numpy.arange(1, self.order + 2).reshape((-1, 1))
    y = numpy.dot(self.Q, p)

    return y


def deriv_impl_constantdenseoutput(self, t: numpy.ndarray):
    '''
    Forward Jacobian implementation for SciPy's ConstantDenseOutput.
    '''
    if t.ndim == 0:
        return numpy.zeros_like(self.value)
    else:
        return numpy.zeros((self.value.shape[0], t.shape[0]))


def deriv_call_denseoutput(self, t):
    t = numpy.asarray(t)
    if t.ndim > 1:
        raise ValueError("`t` must be a float or 1-D array.")
    try:
        return self._deriv_impl(t)
    except AttributeError as exc:
        raise NotImplementedError(f'{type(self)} does not have _deriv_impl.'
                                  ) from exc


def deriv_call_single_odesolution(self, t):
    if self.ascending:
        ind = numpy.searchsorted(self.ts_sorted, t, side='left')
    else:
        ind = numpy.searchsorted(self.ts_sorted, t, side='right')

    segment = min(max(ind - 1, 0), self.n_segments - 1)
    if not self.ascending:
        segment = self.n_segments - 1 - segment

    return self.interpolants[segment].deriv(t)


def deriv_call_odesolution(self, t):
    t = numpy.asarray(t)

    if t.ndim == 0:
        return deriv_call_single_odesolution(self, t)

    order = numpy.argsort(t)
    reverse = numpy.empty_like(order)
    reverse[order] = numpy.arange(order.shape[0])
    t_sorted = t[order]

    # See comment in self._call_single.
    if self.ascending:
        segments = numpy.searchsorted(self.ts_sorted, t_sorted,
                                      side='left')
    else:
        segments = numpy.searchsorted(self.ts_sorted, t_sorted,
                                      side='right')
    segments -= 1
    segments[segments < 0] = 0
    segments[segments > self.n_segments - 1] = self.n_segments - 1
    if not self.ascending:
        segments = self.n_segments - 1 - segments

    ys = []
    group_start = 0
    for segment, group in itertools.groupby(segments):
        group_end = group_start + len(list(group))
        y = self.interpolants[segment].deriv(
            t_sorted[group_start:group_end]
        )
        ys.append(y)
        group_start = group_end

    ys = numpy.hstack(ys)
    ys = ys[..., reverse]

    return ys


def polynomial_impl_rkdenseoutput(self, domain_transform=None
                                  ) -> numpy.ndarray:
    '''
    Return the interpolant as a polynomial.
    '''
    domain = (self.t_old, self.t_old + self.h)
    window = (0.0, 1.0)
    coeff = numpy.concatenate((self.y_old[:, numpy.newaxis], self.h * self.Q),
                              axis=1)

    if domain_transform is not None:
        domain = tuple((domain_transform(t) for t in domain))

    return numpy.array([
        numpy.polynomial.Polynomial(coeff_elem, domain=domain, window=window)
        for coeff_elem in coeff
    ])


def polynomial_call_denseoutput(self, domain_transform=None):
    '''
    Return the interpolant as a polynomial.
    '''
    return self._polynomial_impl(domain_transform)


def polynomial_generator_odesolution(self, domain_transform=None):
    '''
    Iterate over polynomials for each local interpolant.
    '''
    for interpolant in self.interpolants:
        yield interpolant.polynomial(domain_transform)


class MethodInjectionDescriptor:
    '''
    Adds a method to a class.
    '''
    def __init__(self, method: Callable):
        self.func = method

    def __get__(self, obj, objtype=None):
        return functools.partial(self.func, obj)


def register_extensions():
    for klass, method, name in (
        (
            scipy.integrate.OdeSolution,
            deriv_call_odesolution,
            'deriv'
        ),
        (
            scipy.integrate.DenseOutput,
            deriv_call_denseoutput,
            'deriv'
        ),
        (
            scipy.integrate._ivp.rk.RkDenseOutput,
            deriv_impl_rkdenseoutput,
            '_deriv_impl'
        ),
        (
            scipy.integrate._ivp.base.ConstantDenseOutput,
            deriv_impl_constantdenseoutput,
            '_deriv_impl'
        ),
        (
            scipy.integrate.OdeSolution,
            polynomial_generator_odesolution,
            'polynomials'
        ),
        (
            scipy.integrate.DenseOutput,
            polynomial_call_denseoutput,
            'polynomial'
        ),
        (
            scipy.integrate._ivp.rk.RkDenseOutput,
            polynomial_impl_rkdenseoutput,
            '_polynomial_impl'
        ),
    ):
        if not hasattr(klass, name):
            setattr(klass, name, MethodInjectionDescriptor(method))
