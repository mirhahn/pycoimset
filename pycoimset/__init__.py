'''
Python library for continuous optimization with set-valued variables.
'''

from .solver import BarrierSolver, PenaltySolver, UnconstrainedSolver
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
    'BarrierSolver',
    'Constraint',
    'ErrorNorm',
    'Functional',
    'JSONSerializable',
    'Operator',
    'PenaltySolver',
    'UnconstrainedSolver',
    'UnconstrainedStepFinder',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'SteepestDescentStepFinder'
]
