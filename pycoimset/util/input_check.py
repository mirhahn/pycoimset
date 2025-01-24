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
Parameter checking.

This is a basis for a more sophisticated input checking system based on
Python's TypedDict. It is intended to provide runtime type (and value) checking
for parameter dictionaries read from JSON files based purely on type
annotations and decorators attached to the parameter dictionaries.
'''

from collections.abc import Callable, Mapping, Sequence
import inspect
from typing import (
    Any,
    NotRequired,
    Required,
    Type,
    TypeVar,
    get_args,
    get_origin,
    is_typeddict,
)


__all__ = [
    'is_between',
    'is_strictly_between',
    'with_constraints',
    'import_typed_dict',
]


T = TypeVar('T')
D = TypeVar('D', bound=Mapping[Any, Any])


def is_between(key: Any, low: Any, high: Any) -> Callable[[Mapping], None]:
    '''
    Value constraint for fixed value interval.

    Only works for keys whose value supports strict order comparison operators.
    Ignored if the key is not present.

    Arguments
    ---------
    key
        The key used to extract the value from the mapping.

    low
        Lower bound for the value.

    high
        Upper bound for the value.

    Raises
    ------
    ValueError
        The input mapping does not satisfy the constraint.
    '''
    def check(map: Mapping, *, prefix: str = ''):
        if key not in map:
            return
        val = map[key]
        if val < low or val > high:
            raise ValueError(f'{prefix}{key} must be between {low} and {high}')
    return check


def is_strictly_between(key: str, low: Any, high: Any
                        ) -> Callable[[Mapping], None]:
    '''
    Value constraint for fixed open value interval.

    Only works for keys whose value supports non-strict order comparison
    operators. Ignored if the key is not present.

    Arguments
    ---------
    key
        The key used to extract the value from the mapping.

    low
        Strict lower bound for the value.

    high
        Strict upper bound for the value.

    Raises
    ------
    ValueError
        The input mapping does not satisfy the constraint.
    '''
    def check(map: Mapping, *, prefix: str = ''):
        if key not in map:
            return
        val = map[key]
        if val <= low or val >= high:
            raise ValueError(
                f'{prefix}{key} must be strictly between {low} and {high}'
            )
    return check


def with_constraints(constraints: Sequence[Callable[[T], None]]
                     ) -> Callable[[Type[T]], Type[T]]:
    '''
    Decorator to attach value constraints to a type.

    The constraints are stored in the class attribute `value_constraints`.

    Arguments
    ---------
    constraints: sequence of T -> None
        Sequence of constraint checker functions.

    Returns
    -------
    Type[T] -> Type[T]
        The concrete instantiated decorator function.
    '''
    def apply_constraints(type: Type[T]) -> Type[T]:
        setattr(type, 'value_constraints', constraints)
        return type
    return apply_constraints


def import_typed_dict(value: Mapping[Any, Any], dict_type: Type[D], *,
                      allow_unkown: bool = False, prefix: str = '') -> D:
    '''
    Check constraints on an input dictionary and import it.

    This function will recurse into dictionary values if they are annotated
    with typed dictionary types.

    Arguments
    ---------
    value : Mapping
        Input dictionary to be imported.

    dict_type : Type[D]
        Expected type to check against.

    allow_unkown : bool (optional, keyword-only)
        Indicates that unknown keys are allowed to be contained in the
        dictionary. Defaults to `False` to reflect the semantics of
        `TypedDict`.

    prefix : str (optional, keyword-only)
        Name prefix. Used in exception messages.

    Returns
    -------
    D
        Converted shallow copy of the input dictionary.

    Raises
    ------
    KeyError
        A required key is missing or an unkown key is present, but not allowed.

    TypeError
        A value does not have its annotated type.

    Exception
        May pass on other exceptions if they are raised by a value constraint.
    '''
    # Short-circuit for types that are not TypedDict.
    if not is_typeddict(dict_type):
        if isinstance(value, dict_type):
            return value
        return dict_type(value)

    # Make shallow dict of value.
    value = dict(value)

    # Ensure that all required keys are present.
    req_keys = getattr(dict_type, '__required_keys__', set[str]())
    for key in req_keys:
        if key not in value:
            raise KeyError(f'{prefix}{key} is required but not present')

    # Check if there are unknown keys.
    opt_keys = getattr(dict_type, '__optional_keys__', set[str]())
    all_keys = req_keys | opt_keys
    if not allow_unkown:
        for key in value.keys():
            if key not in all_keys:
                raise KeyError(f'{prefix}{key} is present but not defined')

    # Helper for annotation checking.
    def import_annotated(value: Any, name: str, expect: Type[T]) -> T:
        # Unravel Required and NotRequired
        if (orig := get_origin(expect)) is Required or orig is NotRequired:
            subtype, *_ = get_args(expect)
            return import_annotated(value, name, subtype)

        # Check typed sub-dictionaries.
        if issubclass(expect, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f'{prefix}{name} should be a mapping')
            return import_typed_dict(value, expect, allow_unkown=allow_unkown,
                                     prefix='.'.join((prefix, name, '')))

        # Otherwise, we simply use isinstance.
        if not isinstance(value, expect):
            raise TypeError(f'{prefix}{name} should be {expect.__qualname__}')
        return value

    # Check annotations
    annotations = inspect.get_annotations(dict_type)
    for key, annotation in annotations.items():
        if key in value:
            value[key] = import_annotated(value[key], key, annotation)

    # Check constraints
    for check_constr in getattr(dict_type, 'value_constraints', tuple()):
        check_constr(value)

    # Return result
    return dict_type(value)
