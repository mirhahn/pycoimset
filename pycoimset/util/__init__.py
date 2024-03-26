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
Utility functions and types.
'''

from .dependency import depends_on, notify_property_update, tracks_dependencies
from .timing import timed_function
from .weakref import hashref, weak_key_deleter

__all__ = [
    'depends_on',
    'hashref',
    'notify_property_update',
    'timed_function',
    'tracks_dependencies',
    'weak_key_deleter',
]
