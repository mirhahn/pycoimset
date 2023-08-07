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
