'''
Functions and types for error control.
'''

from dataclasses import dataclass


@dataclass
class ErrorBounds:
    '''
    Storage structure for error bounds.
    '''
    errbnd_ob
