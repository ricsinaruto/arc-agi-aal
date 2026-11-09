"""ARC_SEGMENT_TOKEN_CACHE=1 (MFU-1 25 Oct): ships segment_token_runtime.py into the runtime as
vllm/arc_segment_tokens.py and imports it at the CLI entry right after cli_env_setup(), before any tokenizer is built,
so the API server's fast tokenizers encode long prompts piecewise from a cache (exact; see the runtime module). The
flag is read at import per process; off, the import does nothing. An unrecognized main.py keeps vLLM's path
(ARC_SEGMENT_TOKEN_CACHE_PATCH_SKIPPED)."""

import hashlib
import json
from pathlib import Path

TAG = "ARC segment_token_cache"
MAIN = RELATIVE = "vllm/entrypoints/cli/main.py"
RUNTIME = "vllm/arc_segment_tokens.py"
ANCHOR = "    cli_env_setup()\n"
HOOK = ANCHOR + ("    import vllm.arc_segment_tokens as _arc_segment_tokens  # " + TAG + "\n"
                 "    _arc_segment_tokens.install()\n")


def patched_source(original):
    text = original.decode()
    if TAG in text:
        raise RuntimeError("segment_token_cache patch: main.py already carries the hook")
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"segment_token_cache anchor found {text.count(ANCHOR)}x, want 1")
    text = text.replace(ANCHOR, HOOK)
    compile(text, MAIN, "exec")
    return text.encode()


def runtime_source():
    return (Path(__file__).resolve().parent / "segment_token_runtime.py").read_bytes()


def apply(site, output):
    """Never raises: an unrecognized file keeps vLLM's path."""
    target, runtime = Path(site) / MAIN, Path(site) / RUNTIME
    marker = Path(output) / "segment-token-cache-patch.json"
    try:
        original = target.read_bytes()
        if TAG.encode() in original:
            return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
        patched = patched_source(original)
        code = runtime_source()
        compile(code, RUNTIME, "exec")
    except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a boot
        print("ARC_SEGMENT_TOKEN_CACHE_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    runtime.write_bytes(code)
    target.write_bytes(patched)
    identity = {"original_sha256": hashlib.sha256(original).hexdigest(),
                "patched_sha256": hashlib.sha256(patched).hexdigest(),
                "runtime_sha256": hashlib.sha256(code).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    return identity
