"""Decoded-image cache for the API server (``serving_fast_patch`` copies this file to ``vllm/arc_image_cache.py``).

A game request re-sends every history frame as a base64 PNG, and the server decoded each one again (fe2 profile: ~0.3
core over 8 loader threads). Keyed by the payload string and the decode settings, LRU-bounded by decoded size
(4 GiB). A hit returns the same ``MediaWithBytes``: its ``original_bytes`` are this
payload's, so the multimodal hash is unchanged, and the loaders downstream only read it.
"""

import threading
from collections import OrderedDict

LIMIT = 4096 << 20
_LOCK = threading.Lock()
_CACHE = OrderedDict()
_SIZE = [0]
STATS = {"hits": 0, "misses": 0, "evicted": 0}


def _bytes(value):
    try:
        image = value.media
        return image.width * image.height * len(image.getbands()) + len(value.original_bytes)
    except Exception:  # noqa: BLE001
        return 1 << 20


def cached(io, data, load):
    if type(io).__name__ != "ImageMediaIO" or io.kwargs:
        return load()
    key = (data, io.image_mode, io.rgba_background_color)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            STATS["hits"] += 1
            return hit
    value = load()
    size = _bytes(value)
    with _LOCK:
        STATS["misses"] += 1
        if key not in _CACHE and size <= LIMIT:
            _CACHE[key] = value
            _SIZE[0] += size
            while _SIZE[0] > LIMIT:
                _, old = _CACHE.popitem(last=False)
                _SIZE[0] -= _bytes(old)
                STATS["evicted"] += 1
    return value
