'''
Python library for continuous optimization with set-valued variables.
'''

from .solver import UnconstrainedSolver
from .step import SteepestDescentStepFinder
from .typing import (
    Constraint,
    ErrorNorm,
    Functional,
    JSONSerializable,
    Operator,
    SignedMeasure,
    SimilarityClass,
    SimilaritySpace,
    UnconstrainedStepFinder,
)

__all__ = [
    'Constraint',
    'ErrorNorm',
    'Functional',
    'JSONSerializable',
    'Operator',
    'UnconstrainedSolver',
    'UnconstrainedStepFinder',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'SteepestDescentStepFinder'
]
