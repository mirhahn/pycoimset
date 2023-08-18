# PyCoimset: Python library for COntinuous IMprovement of SETs
#
# Copyright 2023 Mirko Hahn
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
Static typing protocols and helpers.
'''

from .functional import Constraint, ErrorNorm, Functional, Operator
from .io import JSONSerializable
from .solver import UnconstrainedStepFinder
from .space import SignedMeasure, SimilarityClass, SimilaritySpace

__all__ = [
    'Constraint',
    'ErrorNorm',
    'Functional',
    'JSONSerializable',
    'Operator',
    'SignedMeasure',
    'SimilarityClass',
    'SimilaritySpace',
    'UnconstrainedStepFinder',
]
