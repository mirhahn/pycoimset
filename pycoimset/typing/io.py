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
Static typing protocols for I/O operations.
'''

import dataclasses
from typing import Protocol, Self, cast


__all__ = ['JSONSerializable']


class JSONSerializable(Protocol):
    def toJSON(self) -> dict | list:
        '''Serialize the object into a JSON-compatible object.'''
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        raise NotImplementedError()

    @classmethod
    def fromJSON(cls, obj: dict | list) -> Self:
        '''Deserialize an object from JSON data.'''
        if isinstance(obj, list):
            return cast(Self, cls(*obj))
        return cast(Self, cls(**obj))
