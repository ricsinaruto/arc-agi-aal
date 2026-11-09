"""The Ornith LoRA test pipeline without GPUs: the M0 builder, the re-quantization helpers, the Modal
launcher's pure functions (resources, reservations, input checks) and the job's schedule helpers."""

import base64
import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

from posttrain import m0, requant

ROOT = Path(__file__).resolve().parents[1]
PAD = "<|image_pad|>"


def png_url(width=256, height=256):
    raw = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return "data:image/png;base64," + base64.b64encode(raw + b"\x00" * 32).decode()


class CharTokenizer:
    """One token per character; the image placeholder is one token (id 7)."""

    image_pad_id = 7
    path = Path("char-tokenizer.json")
    sha256 = "0" * 64

    def encode(self, text):
        ids, offsets, i = [], [], 0
        while i < len(text):
            if text.startswith(PAD, i):
                ids.append(self.image_pad_id)
                offsets.append((i, i + len(PAD)))
                i += len(PAD)
            else:
                ids.append(100 + ord(text[i]) % 50)
                offsets.append((i, i + 1))
                i += 1
        return ids, offsets


def item(item_id="run:ar25-0c556536:L1:C1", source="dev", game_id="ar25-0c556536", level=1, segment=None, repeat=0,
         images=1):
    # prompt (10 chars) + image + target A (5) + 3 chars + image + target B (6)
    text = "sys prompt" + PAD + "AAAAA" + "usr" + PAD * (images - 1) + "BBBBBB"
    a = len("sys prompt" + PAD)
    b = len("sys prompt" + PAD + "AAAAA" + "usr" + PAD * (images - 1))
    return {"id": item_id, "source": source, "format": "packed", "game_id": game_id, "level": level,
            "segment": segment or item_id.rsplit(":", 1)[0], "repeat": repeat, "text": text,
            "images": [png_url()] * images, "loss_spans": [[a, a + 5], [b, b + 6]], "tokens": 0}


def test_tokenize_item_expands_images_and_maps_spans():
    record, reason = m0.tokenize_item(item(images=2), CharTokenizer(), max_tokens=10_000)
    assert reason is None
    ids = record["input_ids"]
    assert ids.count(7) == 2 * m0.IMAGE_TOKENS and len(record["images"]) == 2
    # prompt 10 chars + 64 image positions, then target A
    assert record["loss_spans"][0] == [10 + 64, 10 + 64 + 5]
    assert record["loss_spans"][1] == [79 + 3 + 64, 79 + 3 + 64 + 6]
    assert record["loss_tokens"] == 11 and record["targets"] == 2 and not record["truncated"]
    assert record["straddling_tokens"] == 0 and len(record["sha256"]) == 64


def test_tokenize_item_truncates_after_the_last_whole_target():
    whole = m0.tokenize_item(item(images=2), CharTokenizer(), max_tokens=10_000)[0]
    cut = m0.tokenize_item(item(images=2), CharTokenizer(), max_tokens=whole["loss_spans"][1][1] - 1)[0]
    assert cut["loss_spans"] == whole["loss_spans"][:1] and cut["truncated"] and cut["targets_dropped"] == 1
    assert cut["tokens"] == whole["loss_spans"][0][1] and len(cut["images"]) == 1
    assert m0.tokenize_item(item(), CharTokenizer(), max_tokens=20) == (None, "no whole target inside the window")


def test_tokenize_item_rejects_other_image_sizes_and_counts():
    bad = item()
    bad["images"] = [png_url(512, 512)]
    with pytest.raises(ValueError, match="size"):
        m0.tokenize_item(bad, CharTokenizer(), max_tokens=1000)
    bad = item()
    bad["images"] = []
    with pytest.raises(ValueError, match="placeholders"):
        m0.tokenize_item(bad, CharTokenizer(), max_tokens=1000)


def test_selection_caps_dev_segments_per_game_and_keeps_level_one_first_repeats():
    items = [item(f"run{i}:ar25-0c556536:L1:C1") for i in range(3)]
    items.append(item("run0:ar25-0c556536:L1:C2", segment="run0:ar25-0c556536:L1"))  # second chunk, same segment
    items.append(item("run9:ar25-0c556536:L2:C1", level=2))
    items.append(item("t:q001-aaaa:L1:C1", source="teacher", game_id="q001-aaaaaaaa"))
    items.append(item("t:q001-aaaa:L1:C1", source="teacher", game_id="q001-aaaaaaaa", repeat=1))
    chosen = m0.select_train(items, m0.M0Config(dev_cap_per_game=2))
    assert [i["id"] for i in chosen] == ["run0:ar25-0c556536:L1:C1", "run1:ar25-0c556536:L1:C1",
                                         "run0:ar25-0c556536:L1:C2", "t:q001-aaaa:L1:C1"]
    with pytest.raises(ValueError):
        m0.select_train([item("x:sb26-7fbdac44:L1:C1", game_id="sb26-7fbdac44")], m0.M0Config())
    with pytest.raises(ValueError):
        m0.select_eval([item("x:q001-aaaa:L1:C1", source="teacher", game_id="q001-aaaaaaaa")], m0.M0Config())


def test_build_writes_records_stats_and_manifest(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "train_packed.jsonl").write_text("".join(json.dumps(i) + "\n" for i in (
        item(), item("t:q001-aaaa:L1:C1", source="teacher", game_id="q001-aaaaaaaa"))))
    (data / "eval_packed.jsonl").write_text(json.dumps(item("e:q006-a3e60ccd:L1:C1", source="teacher",
                                                            game_id="q006-a3e60ccd")) + "\n")
    stats = m0.build(tmp_path / "m0", dataset=data, tokenizer=CharTokenizer(), log=lambda *a: None)
    manifest = json.loads((tmp_path / "m0/manifest.json").read_text())
    assert stats["train"]["sequences"] == 2 and stats["eval"]["sequences"] == 1
    assert manifest["files"]["train.jsonl"]["sha256"] == m0.file_sha256(tmp_path / "m0/train.jsonl")
    assert manifest["image_token_id"] == 7 and manifest["image_tokens"] == 64
    rows = [json.loads(line) for line in (tmp_path / "m0/train.jsonl").read_text().splitlines()]
    assert all(r["sha256"] == m0.record_sha256(r) for r in rows)


def test_expected_tokenizer_is_the_served_one():
    served = json.loads((ROOT / "src/arena/models/ornith15-35b-a3b-nvfp4.json").read_text())
    bf16 = json.loads((ROOT / "src/posttrain/inventories/ornith15-35b-a3b-bf16.json").read_text())
    assert m0.expected_tokenizer_sha256() == served["files"]["tokenizer.json"]["sha256"]
    assert bf16["files"]["tokenizer.json"]["sha256"] == served["files"]["tokenizer.json"]["sha256"]
    assert bf16["revision"] == "10fbf86fed7ecee4a061f8b499a618f46001cac1"
    assert sum(n.endswith(".safetensors") for n in bf16["files"]) == 16


# --- requant ---------------------------------------------------------------------------------------


def test_target_modules_formats_and_groups():
    index = json.loads((ROOT / "src/arena/models/ornith15-35b-a3b-nvfp4.json").read_text())
    names = [f"model.language_model.layers.{i}.{m}.weight" for i in (0, 3)
             for m in ("self_attn.q_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_a",
                       "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert_gate",
                       "mlp.experts.gate_up_proj", "mlp.gate")]
    names += ["mtp.layers.0.self_attn.q_proj.weight", "model.visual.blocks.0.attn.qkv.weight"]
    modules = requant.target_modules(names)
    assert modules == [f"model.language_model.layers.{i}.{m}" for i in (0, 3)
                       for m in ("linear_attn.in_proj_qkv", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                                 "self_attn.q_proj")]
    config = {"quantization": {"quantized_layers": {
        **{m: {"quant_algo": "FP8"} for m in modules if "shared_expert" not in m},
        **{m: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for m in modules if "shared_expert" in m}}}}
    formats = requant.quant_formats(config, modules)
    assert formats[modules[0]] == "FP8" and formats[modules[1]] == "W4A16_NVFP4"
    assert requant.scale2_group(modules[1]) == requant.scale2_group(modules[2]) != requant.scale2_group(modules[0])
    assert requant.served_tensor_names(modules[1], "W4A16_NVFP4")[-1].endswith(".weight_scale_2")
    del config["quantization"]["quantized_layers"][modules[0]]
    with pytest.raises(ValueError, match="served quantization"):
        requant.quant_formats(config, modules)
    assert index["files"]["tokenizer.json"]


def test_adapter_key_parsing():
    keys = ["base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.language_model.layers.0.mlp.shared_expert.down_proj.lora_A.default.weight",
            "base_model.model.model.language_model.layers.0.mlp.shared_expert.down_proj.lora_B.default.weight"]
    pairs = requant.adapter_pairs(keys)
    assert set(pairs) == {"model.language_model.layers.3.self_attn.q_proj",
                          "model.language_model.layers.0.mlp.shared_expert.down_proj"}
    with pytest.raises(ValueError, match="both"):
        requant.adapter_pairs(keys[:1])
    with pytest.raises(ValueError, match="unexpected"):
        requant.adapter_pairs(["base_model.model.lm_head.weight"])


def test_fp4_rounding_ties_and_packing():
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.1, 6.0, 7.0, 0.3, -2.9])
    codes = requant.cast_fp4(values)
    # ties go to the even code: 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0
    assert [requant.E2M1_VALUES[c & 7] for c in codes.tolist()[:7]] == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert codes[7] == 8  # a negative value rounding to zero keeps its sign bit
    packed = requant.pack_fp4(codes.reshape(1, -1).to(torch.uint8))
    assert torch.equal(requant.unpack_fp4(packed).abs(), torch.tensor(
        [[0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0, 6.0, 6.0, 0.5, 3.0]]))


def test_requantize_round_trips_within_rounding():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    weights = {"model.language_model.layers.0.self_attn.o_proj": torch.randn(64, 32).to(torch.bfloat16),
               "model.language_model.layers.0.mlp.shared_expert.gate_proj": torch.randn(32, 64).to(torch.bfloat16),
               "model.language_model.layers.0.mlp.shared_expert.up_proj": 3 * torch.randn(32, 64).to(torch.bfloat16)}
    formats = {m: ("W4A16_NVFP4" if "shared" in m else "FP8") for m in weights}
    out = requant.requantize(weights, formats)
    gate, up = (f"model.language_model.layers.0.mlp.shared_expert.{p}_proj" for p in ("gate", "up"))
    assert torch.equal(out[gate + ".weight_scale_2"], out[up + ".weight_scale_2"])  # shared by gate and up
    deq = requant.dequantize_nvfp4(out[up + ".weight"], out[up + ".weight_scale"], out[up + ".weight_scale_2"])
    assert float((deq - weights[up].float()).norm() / weights[up].float().norm()) < 0.15
    o = "model.language_model.layers.0.self_attn.o_proj"
    fp8 = out[o + ".weight"].float() * out[o + ".weight_scale"]
    assert float((fp8 - weights[o].float()).norm() / weights[o].float().norm()) < 0.05
    assert requant.tensor_record(out[o + ".weight_scale"])[:2] == ("F32", [])


# --- the Modal launcher ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def launcher():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_lora_under_test", ROOT / "scripts/modal_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mode_resources_and_bounds(launcher):
    ledger = {"rates_per_second": {"physical_cpu_core": 1.31e-05, "memory_gib": 2.22e-06},
              "volume_rate_usd_per_gib_month": 0.09}
    smoke = launcher.mode_spec("smoke")
    assert smoke["gpu"] == "H200" and smoke["seconds"] <= 900
    rate = 0.001261 + 8 * 1.31e-05 + 64 * 2.22e-06
    assert launcher.bound_usd("smoke", smoke, ledger) == pytest.approx((900 + 60) * rate * 1.15)
    train = launcher.mode_spec("train", 7200)
    assert train["seconds"] == 7200
    assert launcher.bound_usd("train", train, ledger) == pytest.approx(7260 * rate * 1.15)
    fetch = launcher.mode_spec("fetch")
    assert fetch["gpu"] is None and fetch["network"]
    assert launcher.bound_usd("fetch", fetch, ledger) == pytest.approx(
        3660 * (4 * 1.31e-05 + 16 * 2.22e-06) * 1.15 + 96 * 0.09 * 7 / 30)
    for mode, seconds in (("smoke", 1200), ("train", 60), ("train", 5 * 3600), ("bogus", None)):
        with pytest.raises(ValueError):
            launcher.mode_spec(mode, seconds)
    # the whole program fits the LoRA headroom (on 26 September)
    total = sum(launcher.bound_usd(m, launcher.mode_spec(m), ledger) for m in ("fetch", "smoke", "train", "merge"))
    assert total < 30


def test_reservation_rejects_duplicates_and_the_cap(launcher, tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/spend.json"
    path.write_text(json.dumps({"infrastructure_reserve_usd": 2, "runs": [], "cap_usd": 10,
                                "rates_per_second": {"physical_cpu_core": 1.31e-05, "memory_gib": 2.22e-06}}))
    bound = launcher.reserve_lora(tmp_path, "lora-smoke", "smoke", launcher.mode_spec("smoke"))
    entry = json.loads(path.read_text())["runs"][0]
    assert entry["charged_bound_usd"] == pytest.approx(bound, abs=1e-4) and entry["gpu"] == "H200"
    assert entry["kind"] == "lora-smoke" and "LoRA" in entry["authorization"]
    before = path.read_text()
    with pytest.raises(ValueError, match="already reserved"):
        launcher.reserve_lora(tmp_path, "lora-smoke", "smoke", launcher.mode_spec("smoke"))
    with pytest.raises(RuntimeError, match="Spend cap"):
        launcher.reserve_lora(tmp_path, "lora-train", "train", launcher.mode_spec("train"))
    assert path.read_text() == before


def test_input_validation(launcher, tmp_path):
    summary = launcher.validate_inputs(ROOT, "fetch")
    assert summary["bf16"]["shards"] == 16 and summary["nvfp4"]["shards"] == 3
    with pytest.raises(ValueError, match="adapter-run"):
        launcher.validate_inputs(ROOT, "merge")
    records = tmp_path / launcher.RECORDS
    records.mkdir(parents=True)
    for rel in (launcher.NVFP4_INVENTORY, launcher.BF16_INVENTORY):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((ROOT / rel).read_text())
    (records / "modal-lora-train-v1.json").write_text(json.dumps({"mode": "train"}))
    (records / "modal-lora-smoke-v1.json").write_text(json.dumps({"mode": "smoke"}))
    assert launcher.validate_inputs(tmp_path, "merge", adapter_run="lora-train-v1")["adapter"] == \
        "/runs/lora/lora-train-v1/adapter"
    assert launcher.validate_inputs(tmp_path, "merge", adapter_run="lora-train-v1", adapter="adapter-step52")
    with pytest.raises(ValueError, match="not a train run"):
        launcher.validate_inputs(tmp_path, "merge", adapter_run="lora-smoke-v1")
    with pytest.raises(ValueError, match="--adapter"):
        launcher.validate_inputs(tmp_path, "merge", adapter_run="lora-train-v1", adapter="../x")


def test_image_spec_pins_the_training_stack(launcher):
    spec = launcher.image_spec()
    pins = " ".join(spec["torch"]["pins"] + spec["pip"])
    for package in ("torch==", "transformers==5.", "peft==", "flash-linear-attention==", "accelerate=="):
        assert package in pins
    assert all((ROOT / rel).is_file() for rel in spec["files"])


# --- the job's pure helpers ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def job():
    spec = importlib.util.spec_from_file_location("lora_job_under_test", ROOT / "scripts/posttrain/lora_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_and_plan(job):
    cfg = job.TrainConfig()
    per_epoch, total, points = job.plan_steps(103, cfg)
    assert (per_epoch, total, points) == (52, 104, [26, 52, 78, 104])
    lrs = [job.lr_at(s, total, cfg) for s in range(total)]
    assert lrs[0] == pytest.approx(cfg.lr / 8) and lrs[7] == pytest.approx(cfg.lr)
    assert max(lrs) == pytest.approx(cfg.lr) and lrs[-1] == pytest.approx(cfg.lr * 0.1, rel=0.01)
    assert all(a >= b for a, b in zip(lrs[7:], lrs[8:]))
    assert job.epoch_order(10, 0, 0) == job.epoch_order(10, 0, 0) != job.epoch_order(10, 1, 0)
    assert sorted(job.epoch_order(10, 1, 0)) == list(range(10))


def test_truncate_record_and_adapter_scaling(job):
    record = {"id": "x", "input_ids": [1, 2, 7, 7, 3, 4, 5, 6], "loss_spans": [[4, 5], [6, 8]],
              "images": ["a"], "tokens": 8, "loss_tokens": 3}
    cut = job.truncate_record(record, 7, image_token_id=7, image_tokens=2)
    assert cut["input_ids"] == [1, 2, 7, 7, 3] and cut["loss_spans"] == [[4, 5]] and cut["loss_tokens"] == 1
    assert job.truncate_record(record, 3, image_token_id=7, image_tokens=2) is None
    with pytest.raises(ValueError, match="splits an image"):
        job.truncate_record({**record, "loss_spans": [[1, 3]]}, 3, image_token_id=7, image_tokens=2)
    assert job.adapter_scaling({"r": 32, "lora_alpha": 64}) == 2.0
    with pytest.raises(ValueError):
        job.adapter_scaling({"r": 32, "lora_alpha": 64, "use_rslora": True})
    assert math.isclose(job.TrainConfig().alpha / job.TrainConfig().rank, 2.0)


def test_git_blob_id_and_inventory_check(job, tmp_path):
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "data").write_bytes(b"abc")
    import hashlib

    inventory = {"empty": {"bytes": 0, "blob_id": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"},
                 "data": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
    assert job.verify_inventory(tmp_path, inventory) == ["data", "empty"]
    (tmp_path / "data").write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="sha256"):
        job.verify_inventory(tmp_path, inventory)
