'''
Dependency tracking between cached properties.
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
    dep: Dependency
    pred: Optional[Callable[[Any], bool]] = None


def dep_str(dep: Dependency) -> str:
    if isinstance(dep, property):
        for func in (dep.fget, dep.fset, dep.fdel):
            if func is not None:
                return func.__name__
    elif isinstance(dep, functools.cached_property):
        return dep.func.__name__
    return str(dep)


def depends_on(*deps: Dependency | tuple[Dependency, Callable[[Any], bool]]
               ) -> Callable[[functools.cached_property[T]], functools.cached_property[T]]:
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
    raise NotImplementedError()


def tracks_dependencies(cls: Type[T]) -> Type[T]:
        # Build the map of dependents.
        deps_map = {}
        for member in cls.__dict__.values():
            if not isinstance(member, functools.cached_property):
                continue

            try:
                deps = cast(list[QualifiedDependency], getattr(member, 'dependencies'))
                for dep in deps:
                    name = dep_str(dep.dep)
                    deps_map[name] = (*deps_map.get(dep, tuple()), (member, dep.pred))
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
