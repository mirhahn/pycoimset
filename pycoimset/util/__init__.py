'''
Utility functions and types.
'''

from .dependency import depends_on, notify_property_update, tracks_dependencies
from .timing import timed_function
from .weakref import hashref, weak_key_deleter

__all__ = [
    'depends_on',
    'hashref',
    'notify_property_update',
    'timed_function',
    'tracks_dependencies',
    'weak_key_deleter',
]
