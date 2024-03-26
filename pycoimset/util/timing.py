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
'''Utilities for timing and performance measurement.'''

import time
from typing import Callable, Concatenate, Optional, ParamSpec, TypeVar


__all__ = ['timed_function']


P = ParamSpec('P')
R = TypeVar('R')


def timed_function(
    callback: Callable[Concatenate[int, P], None],
    clk_fn: Optional[Callable[[], int]] = None,
    keep_name: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    '''
    Wrap arbitrary function in timing code.

    Parameters
    ----------
    callback : Callable[[int], None]
        Callback used to store timing result.
    clk_fn : Callable[[], int], optional
        Function used to obtain timestamps. Defaults to
        `time.thread_time_ns`.
    keep_name : bool, optional
        Assigns wrapper same name as original function. Defaults to
        `True`.

    Returns
    -------
    Decorator used to wrap functions.
    '''
    if clk_fn is None:
        clk_fn = time.thread_time_ns
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = clk_fn()
            ret = func(*args, **kwargs)
            callback(clk_fn() - start_time, *args, **kwargs)
            return ret
        if keep_name:
            wrapper.__name__ = func.__name__
            wrapper.__qualname__ = func.__qualname__
        return wrapper
    return decorator
