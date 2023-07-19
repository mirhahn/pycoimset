import functools
from typing import TypeVar, cast, Type


T = TypeVar('T')


def depends_on(*deps):
    def dec(func: functools.cached_property):
        if len(deps) > 0:
            try:
                old_deps = getattr(func, 'dependencies')
            except AttributeError:
                old_deps = set()
            setattr(func, 'dependencies', {*old_deps, *deps})
        return func
    return dec


@functools.singledispatch
def notify_property_update(self, prop):
    raise NotImplementedError()


def tracks_dependencies(cls: Type[T]) -> Type[T]:
        # Build the map of dependents.
        deps_map = {}
        for member in cls.__dict__.values():
            if not isinstance(member, functools.cached_property):
                continue

            try:
                deps = getattr(member, 'dependencies')
                for dep in deps:
                    if isinstance(dep, functools.cached_property):
                        dep = str(dep.func.__name__)
                    elif isinstance(dep, property):
                        dep = dep.fget.__name__     # pyright: ignore
                    deps_map[dep] = (*deps_map.get(dep, tuple()), member)
            except AttributeError:
                pass

        # Build update handler.
        def notify_update(self, prop):
            stack = [*deps_map.get(prop, tuple())]
            visited = set()
            while len(stack) > 0:
                child = cast(functools.cached_property, stack.pop())
                if child.attrname is not None and child.func.__name__ not in visited:
                    try:
                        delattr(self, child.attrname)
                        if child.func.__name__ in deps_map:
                            stack.extend(deps_map[child.func.__name__])
                    except AttributeError:
                        pass
                    visited.add(child.func.__name__)
        notify_property_update.register(cls, notify_update)

        return cls

        
