#!/usr/bin/env python3

from pycoimset import Problem, UnconstrainedSolver

import lotka_volterra.ext.scipy as scipy_ext
from lotka_volterra.objective import LotkaObjectiveFunctional
from lotka_volterra.space import IntervalSimilaritySpace


# Treat warnings as errors.
__import__('warnings').simplefilter('error')

# Register interpolant derivative extensions for SciPy.
scipy_ext.register_extensions()

# Define optimization problem
space = IntervalSimilaritySpace((0.0, 12.0))
objective = LotkaObjectiveFunctional(space)
problem = Problem(space, objective)

# Set up solver.
solver = UnconstrainedSolver(problem)
solver.solve()
