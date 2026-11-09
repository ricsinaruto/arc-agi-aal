"""vLLM general plugin: SIGUSR1 prints every thread's stack of the process to the server log (``arc2.l4_serving``
sends it when a start is given up, so a start that hangs after graph capture says where)."""

import faulthandler
import signal
import sys


def register():
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
