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
Caching helpers for functional evaluators.
'''


from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping, Hashable
from typing import TypeVar, TypeVarTuple, cast
from weakref import ref


K = TypeVar('K', bound=Hashable)
V = TypeVar('V')
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
Vs = TypeVarTuple('Vs')


class LRUCache(MutableMapping[K, V]):
    '''
    Least-recently used (LRU) cache.

    Read and write operations count as a key being 'used'.
    '''
    _d: OrderedDict[K, V]
    _s: int

    def __init__(self, max_size: int):
        self._d = OrderedDict[K, V]()
        self._s = max_size

    @property
    def max_size(self) -> int:
        return self._s

    @max_size.setter
    def max_size(self, size: int) -> None:
        self._s = size
        if len(self._d) > self._s:
            self._d.popitem(last=False)

    def __getitem__(self, key: K) -> V:
        val = self._d[key]
        self._d.move_to_end(key, last=True)
        return val

    def __setitem__(self, key: K, val: V) -> None:
        self._d[key] = val
        self._d.move_to_end(key, last=True)
        if len(self._d) > self._s:
            self._d.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        del self._d[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


def error_control_cache(func: Callable[[T, float], tuple[V, float]],
                        *,
                        cache_size: int | None = None
                        ) -> Callable[[T, float], tuple[V, float]]:
    '''
    Wraps an error controlled evaluator with a cache check.

    Arguments
    ---------
    func : (T, float) -> (V, float)
        Original evaluator. Receives an argument and an error bound. Returns
        a return value and an error estimator.

    cache_size : int (optional, keyword-only)
        Maximal size of the cache. Must be non-negative. A value of `0` causes
        no caching to be applied. If omitted or `None`, causes cache to be
        kept indefinitely. Defaults to `None`.

    Returns
    -------
    (T, float) -> (V, float)
        Wrapped evaluator.
    '''
    if cache_size is not None and cache_size <= 0:
        return func

    if cache_size is None:
        cache = dict[ref[T], tuple[V, float]]()
    else:
        cache = LRUCache[ref[T], tuple[V, float]](max_size=cache_size)

    def wipe_cache(arg: ref[T]):
        try:
            del cache[arg]
        except KeyError:
            pass

    def cached_wrapper(arg: T, err_bnd: float) -> tuple[V, float]:
        # Attempt to retrieve from cache.
        try:
            val, err = cache[ref(arg)]
            if err <= err_bnd:
                return val, err
        except KeyError:
            pass

        val, err = func(arg, err_bnd)
        cache[ref(arg, wipe_cache)] = (val, err)
        return val, err
    return cached_wrapper


def cached_method(
    cache_name: str, max_size: int | None = None
) -> Callable[[Callable[[T, *Ts], V]], Callable[[T, *Ts], V]]:
    '''
    Wraps a method in a cache wrapper.

    The cache is stored in a separate attribute of the specified name. This
    can be costly because multiple exceptions may be thrown per invocation.
    As opposed to the standard `cache` decorator, cache is maintained on a
    per-instance basis and does not overlap between instances.

    Arguments
    ---------
    cache_name : str
        Name of the cache variable. If the class uses `__slots__`, then this
        name must be specified as a slot.

    max_size : int (optional)
        Specifies the size of the cache. Can be `None` to indicate infinite
        size, which can be useful if the number of possible arguments is
        finite. The cache structure will be an `LRUCache` if `max_size` is not
        `None`. Otherwise, it will simply be a `dict`.

    Returns
    -------
    ((T, *Ts) -> V) -> ((T, *Ts) -> V)
        Decorator.

    Notes
    -----
        Currently, only positional arguments are supported. This is a deliberate
        choice to avoid situations where outputs for the same arguments are
        cached multiple times based on whether certain arguments are positional
        or keyword arguments.
    '''
    def decorator(func: Callable[[T, *Ts], V]) -> Callable[[T, *Ts], V]:
        '''
        Wraps a given method in a cache wrapper.

        Arguments
        ---------
        func : (T, *Ts) -> V
            Method to be wrapped.

        Returns
        -------
        (T, *Ts) -> V
            Wrapped method.
        '''
        def cache_wrapper(self: T, *args: *Ts) -> V:
            # Retrieve cache
            try:
                cache = cast(MutableMapping[tuple[*Ts], V], getattr(self, cache_name))
            except AttributeError:
                if max_size is not None:
                    cache = LRUCache[tuple[*Ts], V](max_size=max_size)
                else:
                    cache = dict[tuple[*Ts], V]()
                setattr(self, cache_name, cache)

            # Attempt to pull value from cache
            try:
                return cache[args]
            except KeyError:
                val = func(self, *args)
                cache[args] = val
                return val
        cache_wrapper.__name__ = func.__name__
        cache_wrapper.__qualname__ = func.__qualname__
        cache_wrapper.__doc__ = func.__doc__
        return cache_wrapper
    return decorator


def cached_external_property(max_size: int | None = None) -> Callable[[Callable[[T], V]], Callable[[T], V]]:
    '''
    Wraps a function of a single object in a cache wrapper.

    Arguments
    ---------

    max_size : int (optional)
        Specifies the size of the cache. Can be `None` to indicate infinite
        size, which can be useful if the number of possible arguments is
        finite. The cache structure will be an `LRUCache` if `max_size` is not
        `None`. Otherwise, it will simply be a `dict`.

    Returns
    -------
    (T -> V) -> (T -> V)
        Decorator.
    '''
    def decorator(func: Callable[[T], V]) -> Callable[[T], V]:
        '''
        Wraps a given function in a cache wrapper.

        Arguments
        ---------
        func : (T) -> V
            Method to be wrapped.

        Returns
        -------
        (T) -> V
            Wrapped method.
        '''
        if max_size is None:
            cache = dict[ref[T], V]()
        else:
            cache = LRUCache[ref[T], V](max_size=max_size)

        def wipe_cache(key: ref[T]):
            try:
                del cache[key]
            except KeyError:
                pass

        def cache_wrapper(self: T) -> V:
            # Attempt to pull value from cache
            try:
                return cache[ref(self)]
            except KeyError:
                val = func(self)
                cache[ref(self, wipe_cache)] = val
                return val
        return cache_wrapper
    return decorator
