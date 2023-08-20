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
Dependency tracking between cached properties.

Routines
--------
tracks_dependencies
    Class decorator that enables property dependency tracking.

depends_on
    Method decorator that declares a property dependency.

notify_property_update
    Triggers a cache deletion cascade.
'''

import functools
from typing import Any, Callable, NamedTuple, Optional, TypeVar, cast, Type


__all__ = [
    'depends_on',
    'notify_property_update',
    'tracks_dependencies',
]


T = TypeVar('T')
Dependency = str | property | functools.cached_property


class QualifiedDependency(NamedTuple):
    '''
    Internal data structure used to describe a dependency that requires
    (potentially conditional) deletion of cached values.
    '''
    dep: Dependency
    pred: Optional[Callable[[Any], bool]] = None


def dep_str(dep: Dependency) -> str:
    '''
    Transforms a dependency into a string.

    The string should uniquely identify the dependency within the
    tracking object.

    Parameters
    ----------
    dep : Dependency
        A string, property, or cached property that functions as the
        dependence.
    '''
    if isinstance(dep, property):
        for func in (dep.fget, dep.fset, dep.fdel):
            if func is not None:
                return func.__name__
    elif isinstance(dep, functools.cached_property):
        return dep.func.__name__
    return str(dep)


def depends_on(*deps: Dependency | tuple[Dependency, Callable[[Any], bool]]
               ) -> Callable[[functools.cached_property[T]],
                             functools.cached_property[T]]:
    '''
    Decorator declaring a dependency.

    This decorator is attached to cached properties that have a
    dependency. It must be applied after the `cached_property`
    decorator.

    Parameters
    ----------
    *deps : Dependency or tuple of Dependency and callable
        An enumeration of dependencies. If a callable is provided,
        it is invoked upon a change notification with the `self`
        argument and the cached value is only deleted if the
        callable returns `False`.
    '''
    def dec(func: functools.cached_property):
        if len(deps) > 0:
            try:
                dep_lst = getattr(func, 'dependencies')
            except AttributeError:
                dep_lst = []
                setattr(func, 'dependencies', dep_lst)
            dep_lst = cast(list[QualifiedDependency], dep_lst)
            for dep in deps:
                if isinstance(dep, tuple):
                    dep_lst.append(QualifiedDependency(*dep))
                else:
                    dep_lst.append(QualifiedDependency(dep))
        return func
    return dec


@functools.singledispatch
def notify_property_update(self, name):
    '''
    Notify of a property update.

    This can only be invoked on classes that have the `tracks_dependencies`
    decorator.

    Parameters
    ----------
    self
        The self reference of the tracking object.
    name : str
        Name of the property whose value has changed.
    '''
    raise NotImplementedError()


def tracks_dependencies(cls: Type[T]) -> Type[T]:
    '''
    Enables dependency tracking for a given class.

    This defines an override for `notify_property_update`. It does not
    actually modify the class.

    Parameters
    ----------
    cls : Type[T]
        Class to be modified.
    '''
    # Build the map of dependents.
    deps_map = {}
    for member in cls.__dict__.values():
        if not isinstance(member, functools.cached_property):
            continue
        try:
            deps = cast(tuple[QualifiedDependency], getattr(member, 'dependencies'))
            for dep in deps:
                name = dep_str(dep.dep)
                deps_map[name] = (*deps_map.get(name, tuple()), (member, dep.pred))
        except AttributeError:
            pass

    # Build update handler.
    def notify_update(self, prop):
        stack = [*deps_map.get(prop, tuple())]
        visited = set()
        while len(stack) > 0:
            child, pred = cast(tuple[functools.cached_property, Optional[Callable[[Any], bool]]], stack.pop())
            if (
                child.attrname is None or child.func.__name__ in visited
                or child.attrname not in self.__dict__
                or (pred is not None and not pred(self))
            ):
                continue
            try:
                del self.__dict__[child.attrname]
                if child.func.__name__ in deps_map:
                    stack.extend(deps_map[child.func.__name__])
            except KeyError:
                pass
            visited.add(child.func.__name__)
    notify_property_update.register(cls, notify_update)

    return cls
