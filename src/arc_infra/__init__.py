"""Shared infrastructure: configuration, observations, game sessions, fleets and serving."""

import importlib.util as _util
import sys as _sys

# httpcore imports sniffio on every lock / connection operation; without the package each failed import searched every
# sys.path entry (stat calls, each giving up the GIL). A None entry fails the import at once (v7.1).
if "sniffio" not in _sys.modules and _util.find_spec("sniffio") is None:
    _sys.modules["sniffio"] = None


def _bounded_idle_checks():
    """httpcore polls the socket of every idle pooled connection on every request start and close (``has_expired``);
    with a busy GIL each poll waits a switch interval. Poll an idle connection at most once a second (v7.1); a
    keep-alive expiry below the server's 5 s keep-alive (``KEEPALIVE_EXPIRY``) retires idle connections first."""
    import time

    try:
        from httpcore._async.http11 import AsyncHTTP11Connection, HTTPConnectionState
    except ImportError:  # pragma: no cover - another httpcore layout: unchanged
        return
    if getattr(AsyncHTTP11Connection.has_expired, "_arc_bounded", False):
        return
    original = AsyncHTTP11Connection.has_expired

    def has_expired(self):
        now = time.monotonic()
        if self._expire_at is not None and now > self._expire_at:
            return True
        if self._state != HTTPConnectionState.IDLE or now - getattr(self, "_arc_polled", 0.0) < 1.0:
            return False
        self._arc_polled = now
        return original(self)

    has_expired._arc_bounded = True
    AsyncHTTP11Connection.has_expired = has_expired


KEEPALIVE_EXPIRY = 3.0
_bounded_idle_checks()
