'''
Basic forms for the FEM solver.
'''

from skfem.helpers import dot, grad


__all__ = [
    'a', 'a_w', 'L', 'e_int', 'e_fac', 'ge_int', 'ge_fac'
]


def k(w):
    '''
    Diffusivity.

    Parameters
    ----------
    w : array-like
        Control value. Should be between 0 and 1.
    '''
    return 0.1 + 0.9 * w


def dkdw():
    '''
    Derivative of diffusivity with respect to control.
    '''
    return 0.9


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
    return dot(k(w) * grad(u), grad(v)) # type: ignore


def a_w(u, v, dw = 1.0):
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


def ge_int(z, w, dw, f):
    '''
    Interior term of gradient error estimator.

    Parameters
    ----------
    z
        P1 approximation of averaged solution.
    w
        Control function.
    dw
        Control perturbation.
    f
        Source density.

    Remarks
    -------
    This term is symmetric in the sense that the "averaged solution"
    can be either the PDE solution or the adjoint state.
    '''
    return dkdw() / k(w) * dw * f * z


def ge_fac(gy, yh, z, dw, n):
    '''
    Facet term of gradient error estimator.

    Parameters
    ----------
    gy
        Gradient field of higher order approximation.
    yh
        P1 approximation of difference solution.
    z
        P1 approximation of averaged solution.
    dw
        Control perturbation.
    n
        Outer unit normal.
    '''
    return dkdw() * dw * dot(gy - grad(yh), n) * z # type: ignore
