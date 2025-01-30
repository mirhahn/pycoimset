# PyCoimset: Python library for COntinuous IMprovement of SETs
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
Python library for continuous optimization with set-valued variables.
'''

from .solver import PenaltySolver, UnconstrainedSolver
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
    'PenaltySolver',
    'UnconstrainedSolver',
    'UnconstrainedStepFinder',
    'SimilarityClass',
    'SignedMeasure',
    'SimilaritySpace',
    'SteepestDescentStepFinder'
]


if __package__ is not None and '__version__' not in locals():
    import importlib.metadata
    try:
        __version__ = importlib.metadata.version(__package__)
    except importlib.metadata.PackageNotFoundError:
        pass
