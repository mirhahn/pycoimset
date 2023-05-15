'''
Python library for continuous optimization with set-valued variables.
'''

from .problem import Constraint, Problem
from .solver.unconstrained import UnconstrainedSolver
from .step import SteepestDescentStepFinder
from .typing import (
    Functional,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)

__all__ = [
    'Constraint',
    'Functional',
    'Problem',
    'UnconstrainedSolver',
    'UnconstrainedStepFinder',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'SteepestDescentStepFinder'
]
