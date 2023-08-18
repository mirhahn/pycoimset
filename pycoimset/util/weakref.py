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
Weak reference dictionaries with ID-based hashing for unhashable objects.
'''

from functools import cached_property
from typing import Any, Callable, Generic, TypeVar
from weakref import ref
import weakref


__all__ = [
    'hashref',
    'ref',
    'weak_key_deleter',
]


T = TypeVar('T')
K = TypeVar('K')


class idref(weakref.ref[T], Generic[T]):
    '''
    Drop-in replacement for `weakref.ref`.

    This weak reference is always hashable and its hash is the hash
    of the referent's object ID.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def ref_id(self) -> int:
        '''ID of referent.'''
        return id(self())

    def __hash__(self) -> int:
        return hash(self.ref_id)


def hashref(obj: T, *args, **kwargs) -> weakref.ref[T]:
    '''
    Factory function for generating hashable weak references.
    '''
    if hasattr(type(obj), '__hash__') and obj.__hash__ is not None:
        return weakref.ref[T](obj, *args, **kwargs)
    return idref(obj, *args, **kwargs)


def weak_key_deleter(d: dict[weakref.ref[K], Any] | set[weakref.ref[K]]
                     ) -> Callable[[weakref.ref[K]], None]:
    '''
    Create deleter callback for weak key dictionary.
    '''
    if isinstance(d, dict):
        def deleter(key: weakref.ref[K]):
            try:
                del d[key]
            except KeyError:
                pass
    else:
        def deleter(key: weakref.ref[K]):
            try:
                d.remove(key)
            except KeyError:
                pass
    return deleter
