'''
Python library for continuous optimization with set-valued variables.
'''

from .solver import UnconstrainedSolver
from .step import SteepestDescentStepFinder
from .typing import (
    Functional,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)

__all__ = [
    'Functional',
    'UnconstrainedSolver',
    'UnconstrainedStepFinder',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'SteepestDescentStepFinder'
]
