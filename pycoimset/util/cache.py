'''
Caching helpers for functional evaluators.
'''


from collections import OrderedDict
from collections.abc import Iterator, MutableMapping, Hashable
from typing import TypeVar, final


K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


@final
class LRUCache(MutableMapping[K, V]):
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

    def __getitem__(self, key: K) -> V:
        val = self._d[key]
        self._d.move_to_end(key, last=True)
        return val

    def __setitem__(self, key: K, val: V) -> None:
        self._d[key] = val
        self._d.move_to_end(key, last=True)
        while len(self._d) > self._s:
            self._d.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        del self._d[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)
