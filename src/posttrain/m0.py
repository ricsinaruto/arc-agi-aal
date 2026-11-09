"""M0, the LoRA pipeline test set (``docs/posttrain/lora-test-plan.md``, section 4): level-1 packed
sequences from the built pilot dataset (``posttrain.dataset``), tokenized for the trainer.

Selection (from ``train_packed.jsonl``): ``format == "packed"``, ``level == 1``, ``repeat == 0`` (the
trainer's epochs do the repetition), sources ``dev`` (at most ``dev_cap_per_game`` segments per game,
best first as the dataset ranked them) and ``teacher`` (every level-1 segment of the training
mechanics). The held-out set is ``eval_packed.jsonl``'s level-1 teacher packs (the held-out generated
mechanics, ``split.GENERATED_HOLDOUT``), used for the loss before and after training.

Tokenization: the Ornith-1.5 tokenizer (sha256-checked against the served checkpoint's inventory,
``src/arena/models/ornith15-35b-a3b-nvfp4.json``; the BF16 repo's is byte-identical). Each
``<|image_pad|>`` placeholder becomes ``IMAGE_TOKENS`` pad ids (a 256x256 board image: 16x16 patches
merged 2x2, as the processor and vLLM count it; every image is checked to be 256x256), so ``input_ids``
are exactly what the model sees. Loss spans become token ranges ``[start, end)``: a token is a target
token when its first character lies inside a character loss span (reasoning after ``<think>\\n``,
content, tool calls and ``<|im_end|>``); tokens that straddle a span boundary are counted
(``straddling_tokens``; expected 0).

Truncation at ``max_tokens``: a sequence is cut after the last target that ends inside the window;
targets cut by the window lose their loss (they are dropped with everything after them), and a
sequence with no whole target inside the window is dropped. The cut never falls inside an image
(targets end at ``<|im_end|>``).

Output record (one JSON line): ``id``, ``source``, ``game_id``, ``level``, ``segment``, ``input_ids``,
``loss_spans`` (token ranges; the label of position ``t`` is ``input_ids[t]``, predicted from
``t - 1``), ``images`` (data URLs, in order), ``tokens``, ``loss_tokens``, ``targets``,
``targets_dropped``, ``truncated``, ``sha256`` (of the ids, spans and images).
"""

import base64
import collections
import glob
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from .split import check_eval_games, check_training_games, prefix

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "data" / "posttrain" / "dataset"
MODEL_MANIFEST = REPO / "src" / "arena" / "models" / "ornith15-35b-a3b-nvfp4.json"
TOKENIZER_GLOBS = (
    str(Path.home() / ".cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B*/snapshots/*/tokenizer.json"),
)
IMAGE_PAD = "<|image_pad|>"
IMAGE_TOKENS = 64
IMAGE_SIZE = (256, 256)


@dataclass
class M0Config:
    max_tokens: int = 32768
    dev_cap_per_game: int = 8
    sources: tuple = ("dev", "teacher")
    level: int = 1


# --- tokenizer -------------------------------------------------------------------------------------


def expected_tokenizer_sha256(manifest=MODEL_MANIFEST):
    return json.loads(Path(manifest).read_text())["files"]["tokenizer.json"]["sha256"]


def find_tokenizer(path=None):
    """The Ornith tokenizer.json: ``path`` or the first local HF-cache copy."""
    if path:
        return Path(path)
    for pattern in TOKENIZER_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError("Ornith tokenizer.json not found; pass --tokenizer (e.g. hf download "
                            "ornith-ai/Ornith-1.5-35B-A3B tokenizer.json --revision 10fbf86f...)")


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Tokenizer:
    """A ``tokenizers.Tokenizer`` with the sha256 checked against the served checkpoint's inventory."""

    def __init__(self, path=None, *, expected_sha256=None, check=True):
        import tokenizers

        self.path = find_tokenizer(path)
        self.sha256 = file_sha256(self.path)
        expected = expected_sha256 or (expected_tokenizer_sha256() if check else None)
        if expected and self.sha256 != expected:
            raise ValueError(f"{self.path}: tokenizer sha256 {self.sha256} is not the served Ornith tokenizer")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(self.path))
        self.image_pad_id = self.tokenizer.token_to_id(IMAGE_PAD)
        if self.image_pad_id is None:
            raise ValueError(f"{self.path}: no {IMAGE_PAD} token")

    def encode(self, text):
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets


# --- images ----------------------------------------------------------------------------------------


def png_size(data_url):
    """(width, height) of a base64 PNG data URL, read from the IHDR chunk."""
    head, _, payload = data_url.partition(",")
    if not head.startswith("data:image/png;base64"):
        raise ValueError(f"expected a base64 PNG data URL, got {head[:40]!r}")
    raw = base64.b64decode(payload[:64] + "=" * (-len(payload[:64]) % 4))
    if raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", raw[16:24])


# --- one sequence ----------------------------------------------------------------------------------


def tokenize_item(item, tokenizer, *, max_tokens):
    """The training record of one packed dataset item, or ``(None, reason)`` when nothing is left."""
    text, char_spans, images = item["text"], [tuple(s) for s in item["loss_spans"]], list(item["images"])
    if text.count(IMAGE_PAD) != len(images):
        raise ValueError(f"{item['id']}: {text.count(IMAGE_PAD)} image placeholders for {len(images)} images")
    for url in images:
        if png_size(url) != IMAGE_SIZE:
            raise ValueError(f"{item['id']}: image of size {png_size(url)}, expected {IMAGE_SIZE}")
    ids, offsets = tokenizer.encode(text)
    # expand each image placeholder to IMAGE_TOKENS positions (the vision features' slots)
    expanded, starts = [], []
    for token, (start, _end) in zip(ids, offsets):
        repeat = IMAGE_TOKENS if token == tokenizer.image_pad_id else 1
        expanded.extend([token] * repeat)
        starts.extend([start] * repeat)
    if expanded.count(tokenizer.image_pad_id) != IMAGE_TOKENS * len(images):
        raise ValueError(f"{item['id']}: image pad tokens do not match the images")
    # char spans -> token spans (a token belongs to the span that holds its first character)
    token_spans, cursor = [], 0
    for char_start, char_end in sorted(char_spans):
        while cursor < len(starts) and starts[cursor] < char_start:
            cursor += 1
        first = cursor
        while cursor < len(starts) and starts[cursor] < char_end:
            cursor += 1
        if first < cursor:
            token_spans.append([first, cursor])
    # straddling: a token whose characters cross a span boundary
    boundaries = {b for span in char_spans for b in span}
    straddling = sum(1 for s, e in offsets if any(s < b < e for b in boundaries))
    targets = len(token_spans)
    kept = [span for span in token_spans if span[1] <= max_tokens]
    if not kept:
        return None, "no whole target inside the window"
    cut = kept[-1][1]
    input_ids = expanded[:cut]
    n_images = input_ids.count(tokenizer.image_pad_id)
    if n_images % IMAGE_TOKENS:
        raise ValueError(f"{item['id']}: the cut splits an image")
    images = images[:n_images // IMAGE_TOKENS]
    record = {
        "id": item["id"], "source": item["source"], "game_id": item["game_id"], "level": item["level"],
        "segment": item.get("segment"), "input_ids": input_ids, "loss_spans": kept, "images": images,
        "tokens": len(input_ids), "loss_tokens": sum(e - s for s, e in kept), "targets": len(kept),
        "targets_dropped": targets - len(kept), "truncated": len(kept) < targets,
        "untruncated_tokens": len(expanded), "straddling_tokens": straddling,
    }
    record["sha256"] = record_sha256(record)
    return record, None


def record_sha256(record):
    digest = hashlib.sha256()
    digest.update(json.dumps([record["input_ids"], record["loss_spans"]], separators=(",", ":")).encode())
    for url in record["images"]:
        digest.update(url.encode())
    return digest.hexdigest()


# --- selection -------------------------------------------------------------------------------------


def read_items(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_train(items, config):
    """Level-1 packed items of the chosen sources, first repeat only, dev capped per game (in the
    dataset's order, which is best first within a game and level)."""
    chosen, segments_per_game = [], collections.defaultdict(list)
    for item in items:
        if (item.get("format") != "packed" or item.get("level") != config.level or item.get("repeat", 0) != 0
                or item.get("source") not in config.sources):
            continue
        if item["source"] == "dev":
            game = prefix(item["game_id"])
            if item["segment"] not in segments_per_game[game]:
                if len(segments_per_game[game]) >= config.dev_cap_per_game:
                    continue
                segments_per_game[game].append(item["segment"])
        chosen.append(item)
    check_training_games(i["game_id"] for i in chosen)
    return chosen


def select_eval(items, config):
    chosen = [i for i in items if i.get("format") == "packed" and i.get("level") == config.level
              and i.get("source") == "teacher" and i.get("repeat", 0) == 0]
    check_eval_games(i["game_id"] for i in chosen)
    return chosen


def summarize(records, dropped):
    tokens = sorted(r["tokens"] for r in records)
    by_source = collections.Counter(r["source"] for r in records)
    return {
        "sequences": len(records), "tokens": sum(tokens), "loss_tokens": sum(r["loss_tokens"] for r in records),
        "targets": sum(r["targets"] for r in records),
        "targets_dropped_by_window": sum(r["targets_dropped"] for r in records),
        "truncated_sequences": sum(r["truncated"] for r in records),
        "untruncated_tokens": sum(r["untruncated_tokens"] for r in records),
        "images": sum(len(r["images"]) for r in records),
        "straddling_tokens": sum(r["straddling_tokens"] for r in records),
        "max_tokens_seen": tokens[-1] if tokens else 0,
        "median_tokens": tokens[len(tokens) // 2] if tokens else 0,
        "by_source": {s: {"sequences": n, "tokens": sum(r["tokens"] for r in records if r["source"] == s),
                          "loss_tokens": sum(r["loss_tokens"] for r in records if r["source"] == s)}
                      for s, n in sorted(by_source.items())},
        "games": dict(sorted(collections.Counter(prefix(r["game_id"]) for r in records).items())),
        "dropped": dropped,
    }


def build(out_dir, *, dataset=DATASET, tokenizer=None, config=None, log=print):
    """Write ``train.jsonl``, ``eval.jsonl``, ``stats.json`` and ``manifest.json`` into ``out_dir``."""
    config = config or M0Config()
    tokenizer = tokenizer or Tokenizer()
    out, dataset = Path(out_dir), Path(dataset)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"config": asdict(config), "tokenizer": {"path": str(tokenizer.path), "sha256": tokenizer.sha256},
             "image_tokens": IMAGE_TOKENS,
             "inputs": {name: {"path": str(dataset / name), "sha256": file_sha256(dataset / name)}
                        for name in ("train_packed.jsonl", "eval_packed.jsonl", "stats.json")
                        if (dataset / name).is_file()}}
    files = {}
    for split, name, select in (("train", "train_packed.jsonl", select_train), ("eval", "eval_packed.jsonl", select_eval)):
        records, dropped = [], []
        for item in select(read_items(dataset / name), config):
            record, reason = tokenize_item(item, tokenizer, max_tokens=config.max_tokens)
            if record is None:
                dropped.append({"id": item["id"], "reason": reason, "tokens": item.get("tokens")})
            else:
                records.append(record)
        path = out / f"{split}.jsonl"
        with open(path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size,
                            "sequences": len(records)}
        stats[split] = summarize(records, dropped)
        log(f"{split}: {len(records)} sequences, {stats[split]['tokens']:,} tokens, "
            f"{stats[split]['loss_tokens']:,} loss tokens, {len(dropped)} dropped")
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    manifest = {"files": files, "tokenizer_sha256": tokenizer.sha256, "config": asdict(config),
                "image_tokens": IMAGE_TOKENS, "image_token_id": tokenizer.image_pad_id}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return stats
