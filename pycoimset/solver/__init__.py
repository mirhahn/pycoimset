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
"""
Solver implementations.

Classes
-------
PenaltySolver
    Naive quadratic trust-region penalty method for constrained problems.
    May yield points slightly outside the feasible region.

UnconstrainedSolver
    Simple trust-region steepest descent method for unconstrained
    problems.
"""
from .penalty import PenaltySolver
from .unconstrained import UnconstrainedSolver

__all__ = [
    'PenaltySolver',
    'UnconstrainedSolver',
]
