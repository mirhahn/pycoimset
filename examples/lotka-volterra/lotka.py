#!/usr/bin/env python3

import numpy
import scipy.integrate
from lotka_volterra.ode import LotkaAdjointErrorIVP, LotkaAdjointIVP, \
                               LotkaForwardErrorIVP, LotkaForwardIVP, \
                               LotkaGradientDensity, \
                               LotkaGradientDensityErrorIVP
from lotka_volterra.polyfit import merge_time_grids, midpoint_time_grid, \
                                   polyfit_quartic

import lotka_volterra.ext.scipy


lotka_volterra.ext.scipy.register_extensions()

fwd_ivp = LotkaForwardIVP([0.4, 0.2], [0.5, 0.7], 0, 12)
fwd_err_ivp = LotkaForwardErrorIVP(fwd_ivp)
adj_ivp = LotkaAdjointIVP(fwd_ivp)
adj_err_ivp = LotkaAdjointErrorIVP(adj_ivp)
grad_eval = LotkaGradientDensity(adj_ivp)
grad_err_ivp = LotkaGradientDensityErrorIVP(grad_eval)

result_fwd = scipy.integrate.solve_ivp(
    fwd_ivp, fwd_ivp.time_range, fwd_ivp.initial_state, dense_output=True,
    method='RK45', atol=1e-10, rtol=1e-10, args=(0.0,)
)
fwd_err_ivp.fwd_sol = result_fwd.sol
result_fwderr = scipy.integrate.solve_ivp(
    fwd_err_ivp, fwd_err_ivp.time_range, fwd_err_ivp.initial_state,
    dense_output=True, method='RK45', atol=1e-10, rtol=1e-10, args=(0.0,)
)
adj_ivp.fwd_sol = result_fwd.sol
result_adj = scipy.integrate.solve_ivp(
    adj_ivp, adj_ivp.time_range, adj_ivp.initial_state, dense_output=True,
    method='RK45', atol=1e-10, rtol=1e-10, args=(0.0,)
)
adj_err_ivp.fwd_sol = result_fwd.sol
adj_err_ivp.fwd_err_sol = result_fwderr.sol
adj_err_ivp.adj_sol = result_adj.sol
result_adjerr = scipy.integrate.solve_ivp(
    adj_err_ivp, adj_err_ivp.time_range, adj_err_ivp.initial_state,
    dense_output=True, method='RK45', atol=1e-10, rtol=1e-10, args=(0.0,)
)

# Fit an approximation spline for the (unsigned) gradient density function.
time_grid = merge_time_grids(
    result_fwd.t, numpy.flip(adj_ivp.export_times(result_adj.t))
)
midpoint_grid = midpoint_time_grid(time_grid)
grad_eval.fwd_sol = result_fwd.sol
grad_eval.fwd_err_sol = result_fwderr.sol
grad_eval.adj_sol = result_adj.sol
grad_eval.adj_err_sol = result_adjerr.sol
grad_sol = polyfit_quartic(
    midpoint_grid, grad_eval(midpoint_grid), grad_eval.deriv(time_grid)
)

# Estimate gradient density error.
grad_err_ivp.grad_dens_traj = grad_sol
result_graderr = scipy.integrate.solve_ivp(
    grad_err_ivp, grad_err_ivp.time_range, grad_err_ivp.initial_state,
    dense_output=True, method='RK45', atol=1e-10, rtol=1e-10, args=(0.0,)
)
