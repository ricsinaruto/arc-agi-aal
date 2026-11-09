"""Count chat inputs through the pinned SGLang generation preprocessor.

Only apply to a disposable extracted runtime. The checkpoint and frozen runtime
archive are untouched. This performs multimodal preprocessing, not generation.
"""

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

TARGET = Path("python/sglang/srt/entrypoints/openai/serving_tokenize.py")
ORIGINAL_SHA256 = "f1791dbe89245cf80f2ae3c3f052e944c0b602deaf45909a31d2e4aabfc95711"


async def _tokenize_chat_request(self, request):
    """Use the same template and expanded visual tokens as chat generation."""
    if self.chat_serving is None:
        raise ValueError("Chat template tokenization requires a template manager.")
    manager = self.tokenizer_manager
    # Never silently count a shortened prompt, or mutate a shared manager flag.
    if manager.allow_auto_truncate:
        raise ValueError("Exact chat counting requires auto truncation disabled.")
    chat_request = request.to_chat_completion_request()
    validation_error = self.chat_serving._validate_request(chat_request)
    if validation_error:
        raise ValueError(validation_error)
    adapted, _ = self.chat_serving._convert_to_internal_request(chat_request)
    if adapted.lora_path:
        raise ValueError("Exact arena chat counting does not support LoRA adapters.")
    # No generation is submitted. A zero allowance avoids validating a default
    # completion length as part of this input-only request.
    adapted.sampling_params["max_new_tokens"] = 0
    adapted.normalize_batch_and_arguments()
    if not adapted.is_single:
        raise ValueError("Exact arena chat counting requires a single conversation.")
    # The pinned manager's object constructor attaches timing from rid_to_state.
    # Register the same local state generation uses, but never dispatch to the
    # scheduler. Initialization stays outside try so duplicate-ID rejection cannot
    # remove an existing live request. Release on success, error or cancellation.
    manager._init_req_state(adapted, None)
    try:
        tokenized = await manager._tokenize_one_request(adapted)
        if tokenized.input_ids is None:
            raise ValueError("Chat preprocessing returned no token IDs.")
        return list(tokenized.input_ids)
    finally:
        manager._release_req_states_on_failure({adapted.rid})


def patched_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("SGLang tokenize source differs from the pinned runtime")
    source = original.decode()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "OpenAIServingTokenize")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_tokenize_chat_request")
    lines = source.splitlines(keepends=True)
    lines[method.lineno - 1:method.end_lineno] = [
        textwrap.indent(inspect.getsource(_tokenize_chat_request), "    ")
    ]
    source = "".join(lines)
    old = "token_ids = self._tokenize_chat_request(request)"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected SGLang tokenize call site")
    source = source.replace(old, "token_ids = await self._tokenize_chat_request(request)")
    compile(source, str(TARGET), "exec")
    return source.encode()


def apply(repo_root: Path | str) -> dict:
    target = Path(repo_root) / TARGET
    backup = target.with_suffix(".py.arc-original")
    original = backup.read_bytes() if backup.exists() else target.read_bytes()
    patched = patched_source(original)
    current = target.read_bytes()
    if current not in (original, patched):
        raise RuntimeError("SGLang tokenize file has unrelated modifications")
    if not backup.exists():
        backup.write_bytes(original)
    if current != patched:
        temporary = target.with_suffix(".py.arc-tmp")
        temporary.write_bytes(patched)
        temporary.replace(target)
    return {"original_sha256": ORIGINAL_SHA256,
            "patched_sha256": hashlib.sha256(patched).hexdigest(),
            "path": str(target), "mode": "generation_multimodal_preprocessing"}
