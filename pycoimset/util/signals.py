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
Helpers to intercept and defer user interruption signals.
'''

import contextlib
import signal


__interrupt: bool = False
'''
Global flag indicating that a SIGINT has been intercepted.
'''


def interrupt_requested() -> bool:
    '''Check whether a global interrupt request has been made.'''
    global __interrupt
    return __interrupt


def request_interrupt() -> None:
    '''Set the global interrupt request flag.'''
    global __interrupt
    __interrupt = True


def setup_interrupt() -> None:
    '''Set up a signal handler for SIGINT and SIGTERM.'''
    def handler(*_):
        request_interrupt()
    signal.signal(signal.SIGINT | signal.SIGTERM, handler)


@contextlib.contextmanager
def defer_sigint():
    '''Defer SIGINT until end of block.'''
    # Set up dummy handler.
    signal_received = None
    def handler(*args):
        signal_received = args

    # Install dummy handler.
    old_handler = signal.signal(signal.SIGINT, handler)

    # Yield to code block.
    yield

    # Replace handler.
    signal.signal(signal.SIGINT, old_handler)
    if signal_received is not None:
        signal.raise_signal(signal.SIGINT)