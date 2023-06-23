import signal


#: Global flag for SIGINT or SIGTERM.
__interrupt: bool = False


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
