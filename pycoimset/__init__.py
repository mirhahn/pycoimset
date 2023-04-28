'''
Python library for continuous optimization with set-valued variables.
'''

from .functional import Functional
from .problem import Constraint, MappedFunctional, Problem
from .space import SimilarityClass, SignedMeasure

__all__ = [
    'Constraint',
    'Functional',
    'MappedFunctional',
    'Problem',
    'SimilarityClass',
    'SignedMeasure'
]
