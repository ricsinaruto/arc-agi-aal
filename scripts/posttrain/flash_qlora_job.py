"""Flash loader and cached-prefix helpers (post-training pilot, 7 October) used by ``rl1_trainer.py`` in the pinned
Axolotl image's Python (``/workspace/axolotl-venv``): the model loaded as ``axolotl train`` would (NVFP4 experts via
SonicMoE, n-gram table in host RAM, LoRA on attention, Gated DeltaNet and the shared expert), the served request
encoder (``encode_request``, ``example_maker``), positions and images per cached-prefix piece, and the recorder.
Every event is one JSON line on stdout and in ``<out>/events.jsonl``."""

import json
import os
import resource
import threading
import time
from pathlib import Path

TARGET_MODULES = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
                  "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
                  "q_proj", "k_proj", "v_proj", "o_proj"]
GROUPS = (("gdn", "linear_attn."), ("attention", "self_attn."), ("shared_expert", "shared_expert."))
START = time.time()
# RL C2 (rl1_dist): under RL1_DIST=1 (torchrun) each process loads onto its own card; 0 otherwise
CARD_INDEX = int(os.environ.get("LOCAL_RANK", "0")) if os.environ.get("RL1_DIST") == "1" else 0


class Recorder:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.result = {"events": []}

    def event(self, kind, **data):
        row = {"t": round(time.time() - START, 1), "event": kind, **data}
        print(json.dumps(row), flush=True)
        with (self.out / "events.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        return row

    def save(self, **data):
        self.result.update(data)
        (self.out / "result.json").write_text(json.dumps({k: v for k, v in self.result.items() if k != "events"},
                                                         indent=1))


def group_of(name):
    return next((group for group, mark in GROUPS if mark in name), "other")


def config(model_dir, out, *, sequence_len):
    """The Axolotl config: the served NVFP4 checkpoint itself (Axolotl's nvfp4-lora.yaml). The routed
    experts stay NVFP4 (SonicMoE, W4A16 like the serving kernels) with merge-aware adapters through the quantizer;
    everything else is its BF16 weight. No cut cross-entropy plugin: the loss reads logits at the trained tokens only.
    ple_cpu_offload is off (Axolotl allows it only with bitsandbytes); keep_table_in_host_ram() leaves the n-gram
    table in host RAM at load."""
    return {
        "base_model": str(model_dir), "processor_type": "AutoProcessor",
        "plugins": ["axolotl.integrations.kernels.KernelsPlugin"], "ple_cpu_offload": False,
        "skip_prepare_dataset": True, "remove_unused_columns": False, "sample_packing": False,
        "datasets": [{"path": str(Path(out) / "unused.jsonl"), "type": "chat_template"}],
        "val_set_size": 0.0, "output_dir": str(Path(out) / "axolotl"),
        "adapter": "lora", "sequence_len": sequence_len, "pad_to_sequence_len": False,
        "lora_r": ADAPTER["rank"], "lora_alpha": ADAPTER["alpha"], "peft_use_rslora": False,
        "lora_dropout": 0, "lora_target_modules": TARGET_MODULES,
        "lora_mlp_kernel": False, "lora_qkv_kernel": False, "lora_o_kernel": False,
        "gradient_accumulation_steps": 1, "micro_batch_size": 1, "num_epochs": 1, "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine", "learning_rate": 2e-5, "bf16": "auto", "tf32": True,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "attn_implementation": "sdpa", "warmup_ratio": 0.05, "saves_per_epoch": 1, "weight_decay": 0.0,
        "strict": False, "device_map": {"": CARD_INDEX}, "use_sonicmoe": True, "nvfp4_merge_aware": True,
        "lora_target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]}


def set_adapter(rank):
    """Rank of the adapter; alpha = 2 x rank (the update scaled by 2)."""
    ADAPTER.update(rank=rank, alpha=2 * rank)


def host_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)


def rss_split(path="/proc/self/status"):
    """GiB: VmHWM (peak RSS), VmRSS and its RssAnon / RssFile parts; {} where unreadable."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                out[key] = round(int(rest.split()[0]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


def read_bytes():
    try:
        return int(next(line for line in Path("/proc/self/io").read_text().splitlines()
                        if line.startswith("rchar")).split()[1])
    except Exception:
        return 0


def heartbeat(rec, stop, label, seconds=30):
    import torch

    def run():
        last = read_bytes()
        while not stop.wait(seconds):
            now = read_bytes()
            gpu = round(torch.cuda.memory_allocated() / 2 ** 30, 1) if torch.cuda.is_initialized() else None  # a gate
            rec.event("heartbeat", phase=label, gpu_gib=gpu,
                      host_peak_gib=host_gib(), read_gib=round(now / 2 ** 30, 1),
                      read_gib_s=round((now - last) / 2 ** 30 / seconds, 2))
            last = now

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def environment(rec):
    import importlib

    import torch

    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "axolotl", "fla", "cut_cross_entropy",
                 "torchvision", "triton"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "present")
        except Exception as error:  # the report is the point
            versions[name] = f"MISSING {type(error).__name__}: {error}"
    importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    gpu = None
    if preload_wanted() and not torch.cuda.is_initialized():  # the card waits for the gate; no CUDA before the load
        gpu = {"deferred": "RL1_PRELOAD_HOST"}
    elif torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "total_gib": round(total / 2 ** 30, 1),
               "free_gib": round(free / 2 ** 30, 1), "capability": list(torch.cuda.get_device_capability(0))}
    memory = {line.split(":")[0]: int(line.split()[1]) // 2 ** 20 for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal", "MemAvailable"))}
    return rec.event("environment", versions=versions, cuda=torch.cuda.is_available(), gpu=gpu, host_gib=memory,
                     cpus=os.cpu_count())


def keep_table_in_host_ram():
    """transformers joins the n-gram table's shards on CPU and then moves the result to the device map's device.
    Axolotl's single-GPU map makes that the GPU (check v2: the table on cuda:0 and Axolotl's warning that
    ``ple_cpu_offload`` found nothing in host RAM), which is 95 GiB a B200 does not have during the load."""
    import transformers.core_model_loading as loading

    original = loading.get_device
    if getattr(original, "table_in_host_ram", False):
        return

    def get_device(device_map, name, *args, **kwargs):
        if "ngram_embedding." in name:
            return "cpu"
        return original(device_map, name, *args, **kwargs)

    get_device.table_in_host_ram = True
    loading.get_device = get_device


def reset_peaks():
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


PRELOAD_ENV = "RL1_PRELOAD_HOST"  # with RL1_GPU_GATE: the whole model built in host RAM before the gate, then moved
PRELOAD_DEVICE = None  # the move's target (None: cuda:CARD_INDEX); the CPU leg moves to "meta"


PRELOAD_NEED_GIB = 400.0  # per trainer process: host RSS peak of a preloaded load (CPU legs 26 Oct: 385-401 GiB)
# RL1_SHARED_TABLE (rl1_table): per process without the n-gram table, plus the one shared table (51.2e9 BF16). 2-rank
# CPU leg 26 Oct: cgroup peak 493.9 GiB (page cache included) = 95.4 + 2 x 199.3; AnonPages (with Shmem) 422.1
PRELOAD_NEED_SHARED_GIB = 200.0
PRELOAD_TABLE_GIB = 95.4
PRELOAD = {}  # this process's decision (preload_wanted() without arguments): asked, on, room


def host_memory(meminfo="/proc/meminfo", cgroup="/sys/fs/cgroup"):
    """GiB: MemTotal, MemAvailable and the cgroup's limit and usage (v2, else v1); None where unreadable or
    unlimited."""
    out = {"mem_total_gib": None, "mem_available_gib": None, "cgroup_max_gib": None, "cgroup_current_gib": None}
    try:
        for line in Path(meminfo).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[{"MemTotal": "mem_total_gib", "MemAvailable": "mem_available_gib"}[key]] = round(
                    int(rest.split()[0]) / 2 ** 20, 1)
    except (OSError, ValueError, IndexError):
        pass
    for limit, usage in (("memory.max", "memory.current"), ("memory/memory.limit_in_bytes",
                                                            "memory/memory.usage_in_bytes")):
        try:
            top = Path(cgroup, limit).read_text().strip()
            used = int(Path(cgroup, usage).read_text().strip())
        except (OSError, ValueError):
            continue
        out["cgroup_current_gib"] = round(used / 2 ** 30, 1)
        if top.isdigit() and int(top) < 2 ** 60:  # v1 reports "unlimited" as a huge number
            out["cgroup_max_gib"] = round(int(top) / 2 ** 30, 1)
        break
    return out


def preload_room(environ=None, memory=None):
    """Whether this process's host-built model fits: need = RL1_PRELOAD_NEED_GIB (PRELOAD_NEED_GIB) per trainer
    process x the processes starting together (LOCAL_WORLD_SIZE) + 5%, against MemAvailable and the cgroup's
    headroom (the smaller). Unknown room: no fit. RL1_SHARED_TABLE: PRELOAD_NEED_SHARED_GIB per process plus the one
    table (PRELOAD_TABLE_GIB)."""
    environ = os.environ if environ is None else environ
    memory = host_memory() if memory is None else memory
    ranks = int(environ.get("LOCAL_WORLD_SIZE") or 1)
    shared = environ.get("RL1_SHARED_TABLE") == "1"
    per = float(environ.get("RL1_PRELOAD_NEED_GIB") or (PRELOAD_NEED_SHARED_GIB if shared else PRELOAD_NEED_GIB))
    # RL1_PRELOAD_RESERVE_GIB: host memory others will still take (a trainer pre-spawned at container start: the
    # servers' boot, arena.trainer_prespawn)
    need = (per * ranks + (PRELOAD_TABLE_GIB if shared else 0.0)
            + float(environ.get("RL1_PRELOAD_RESERVE_GIB") or 0)) * 1.05
    rooms = [memory.get("mem_available_gib")]
    if memory.get("cgroup_max_gib") is not None and memory.get("cgroup_current_gib") is not None:
        rooms.append(memory["cgroup_max_gib"] - memory["cgroup_current_gib"])
    rooms = [r for r in rooms if r is not None]
    room = min(rooms) if rooms else None
    return {"need_gib": round(need, 1), "ranks": ranks, "shared_table": shared,
            "room_gib": None if room is None else round(room, 1),
            "ok": room is not None and room >= need, "memory": memory}


def preload_wanted(environ=None):
    """RL1_PRELOAD_HOST asked (with a gate). Without ``environ``: this process's decision, made once: asked and the
    host has room (preload_room); else the direct load (H2's path) and a ``preload_skipped`` event."""
    if environ is not None:
        return environ.get(PRELOAD_ENV) == "1" and bool(environ.get("RL1_GPU_GATE"))
    if "on" not in PRELOAD:
        asked = preload_wanted(os.environ)
        PRELOAD.update(asked=asked, on=asked)
        if asked:
            PRELOAD["room"] = room = preload_room()
            if not room["ok"]:
                PRELOAD["on"] = False
                print(json.dumps({"event": "preload_skipped", **room}), flush=True)
    return PRELOAD["on"]


def host_load_device():
    """RL1_PRELOAD_HOST: every tensor transformers loads lands on the host (the n-gram table already does,
    keep_table_in_host_ram); the move to the card waits for the gate (to_card)."""
    import transformers.core_model_loading as loading

    original = loading.get_device
    if getattr(original, "host_preload", False):
        return

    def get_device(device_map, name, *args, **kwargs):
        return "cpu"

    get_device.host_preload = get_device.table_in_host_ram = True
    loading.get_device = get_device


def card_info(environ=None):
    """{major, minor, name, total_memory} of the trainer's card without a CUDA context (NVML), or None.
    RL1_PRELOAD_CAPABILITY ("10.0") stands in for NVML (the CPU leg)."""
    environ = os.environ if environ is None else environ
    if environ.get("RL1_PRELOAD_CAPABILITY"):
        major, minor = environ["RL1_PRELOAD_CAPABILITY"].split(".")
        return {"major": int(major), "minor": int(minor), "name": "preload-shim", "total_memory": 0}
    try:
        import pynvml

        pynvml.nvmlInit()
        index = int((environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {"major": int(major), "minor": int(minor), "name": name.decode() if isinstance(name, bytes) else name,
                "total_memory": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)}
    except Exception:  # noqa: BLE001 - no NVML: no shim (the gate then blocks at the first query, as before)
        return None


def peer_access(device, peer, environ=None):
    """NVML's P2P read status between two visible cards, without a CUDA context (2-rank leg 26 Oct: Axolotl's
    load_cfg asks ``can_device_access_peer`` when WORLD_SIZE > 1, which initialised CUDA, i.e. waited for the gate,
    before the host load). RL1_PRELOAD_CAPABILITY (the CPU leg) or an unreadable status: True (NCCL still probes the
    topology itself; False would make Axolotl set NCCL_P2P_DISABLE=1)."""
    environ = os.environ if environ is None else environ
    if environ.get("RL1_PRELOAD_CAPABILITY"):
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        visible = [x.strip() for x in (environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]

        def handle(d):
            d = int(d)
            if visible and d < len(visible) and not visible[d].isdigit():
                return pynvml.nvmlDeviceGetHandleByUUID(visible[d])
            return pynvml.nvmlDeviceGetHandleByIndex(int(visible[d]) if visible and d < len(visible) else d)

        status = pynvml.nvmlDeviceGetP2PStatus(handle(device), handle(peer), pynvml.NVML_P2P_CAPS_INDEX_READ)
        return status == pynvml.NVML_P2P_STATUS_OK
    except Exception:  # noqa: BLE001
        return True


PRELOAD_SHIM = {}
SHIMMED = ("current_device", "get_device_capability", "get_device_properties", "get_device_name",
           "reset_peak_memory_stats", "can_device_access_peer")


def preload_shims(info=None):
    """Pre-gate answers to the device queries imports and config make (H attempt 2: torchao's import-time
    ``has_triton()`` / ``is_sm_at_least_100()`` asked the card's properties, initialised CUDA and so waited for the gate
    before any of the load ran). Until ``drop_preload_shims`` (after the gate) ``torch.cuda``'s current_device,
    capability, properties and name answer from NVML; afterwards the real functions are back and Inductor's property
    cache, if filled meanwhile, is cleared. Cached answers (``has_triton``, torchao's sm checks) equal the real card's.
    Returns the shim's info, or None (no NVML or already installed)."""
    info = info or card_info()
    if info is None or PRELOAD_SHIM:
        return None
    from types import SimpleNamespace

    import torch
    import torch._dynamo.device_interface as di

    props = SimpleNamespace(**info, is_integrated=False)
    PRELOAD_SHIM.update(info=info, real={name: getattr(torch.cuda, name) for name in SHIMMED}, module=di,
                        cached={k for k in ("cuda",) if k in di.caching_worker_device_properties})
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (info["major"], info["minor"])
    torch.cuda.get_device_properties = lambda device=None: props
    torch.cuda.get_device_name = lambda device=None: info["name"]
    reset = PRELOAD_SHIM["real"]["reset_peak_memory_stats"]  # Axolotl's memory logs reset the peaks: nothing to reset
    torch.cuda.reset_peak_memory_stats = lambda device=None: reset(device) if torch.cuda.is_initialized() else None
    torch.cuda.can_device_access_peer = peer_access  # Axolotl's load_cfg under WORLD_SIZE > 1
    return info


def drop_preload_shims():
    if not PRELOAD_SHIM:
        return None
    import torch

    for name, real in PRELOAD_SHIM.pop("real").items():
        setattr(torch.cuda, name, real)
    di = PRELOAD_SHIM.pop("module")
    if "cuda" not in PRELOAD_SHIM.pop("cached"):
        di.caching_worker_device_properties.pop("cuda", None)
        di.caching_worker_current_devices.pop("cuda", None)
    return PRELOAD_SHIM.pop("info")


def wait_gate(path, poll=1.0):
    began = time.time()
    while not os.path.exists(path):
        time.sleep(poll)
    return round(time.time() - began, 1)


def stored_bytes(t):
    """Bytes a tensor holds: a wrapper subclass (the experts' NVFP4Tensor reports BF16 numel) counts its inner tensors."""
    if hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        return sum(stored_bytes(getattr(t, name)) for name in names if getattr(t, name) is not None)
    return int(t.numel()) * int(t.element_size())  # int(): a subclass may answer with 0-dim tensors


def load_digest(model, part=".layers.3."):
    """sha256 over the bytes (inner tensors of wrapper subclasses too), dtypes and shapes of every parameter and buffer
    named with ``part`` (one decoder layer: NVFP4 experts, attention, LoRA init), wherever they sit."""
    import hashlib

    import torch

    def leaves(t):
        if hasattr(t, "__tensor_flatten__"):
            names, ctx = t.__tensor_flatten__()
            return [(f"{n}", x) for n in names for _, x in leaves(getattr(t, n))] + [("ctx", str(ctx))]
        return [("", t)]

    digest, count = hashlib.sha256(), 0
    for name, t in sorted([*model.named_parameters(), *model.named_buffers()], key=lambda item: item[0]):
        if part not in name or "lora_" in name:  # LoRA A is random at init
            continue
        for sub, leaf in leaves(t.detach()):
            digest.update(f"{name}/{sub}".encode())
            if isinstance(leaf, str):
                digest.update(leaf.encode())
                continue
            leaf = leaf.contiguous().cpu()
            digest.update(f"{leaf.dtype}{list(leaf.shape)}".encode())
            digest.update(leaf.reshape(-1).view(torch.uint8).numpy().tobytes() if leaf.numel() else b"")
        count += 1
    return {"part": part, "tensors": count, "sha256": digest.hexdigest()[:16]}


def release_host():
    """After the preloaded model moved to the card: the host copy's freed blocks back to the OS (real-v10 28 Oct: each
    rank kept 300-330 GiB RSS after the move, against 176 / 21 GiB on a direct load, so play started ~600 GiB short).
    gc first (cycles holding host tensors), then glibc malloc_trim(0) (freed arena chunks stay mapped without it).
    Returns {before, after} rss_split GiB; never raises."""
    import gc

    out = {"before": rss_split()}
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
        out["trimmed"] = True
    except (OSError, AttributeError):  # no glibc (macOS tests)
        out["trimmed"] = False
    out["after"] = rss_split()
    return out


def to_card(model, device, keep=lambda name: "ngram_embedding." in name):
    """Every parameter and buffer of ``model`` to ``device`` except the ``keep`` ones (the n-gram table stays in host
    RAM, as at a direct load), module by module through ``Module._apply`` (tensor subclasses swap as in ``.to``).
    Returns the bytes moved."""
    import torch

    moved = [0]
    for prefix, module in model.named_modules():
        skip = {id(t) for n, t in [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]
                if keep(f"{prefix}.{n}" if prefix else n)}

        def fn(t):
            if id(t) in skip or t.device == torch.device(device):
                return t
            moved[0] += stored_bytes(t)
            return t.to(device)

        module._apply(fn, recurse=False)
    return moved[0]


def load(rec, model_dir, out, *, sequence_len, before_gate=None, table_meta=False):
    import torch
    import yaml
    from axolotl.cli.config import load_cfg
    from axolotl.train import setup_model_and_tokenizer

    preload = preload_wanted()
    spec = config(model_dir, out, sequence_len=sequence_len)
    if preload:  # H attempt 2 (26 Oct): the gated load took 305 s after the gate; the host part now runs before it
        spec["device_map"] = {"": "cpu"}
    path = Path(out) / "config.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    cfg = load_cfg(str(path))
    if preload:  # Axolotl's normalisation under WORLD_SIZE > 1 sets {"": LOCAL_RANK}; transformers then put the
        cfg.device_map = {"": "cpu"}  # non-persistent buffers on the card before the gate (2-rank CPU leg 26 Oct)
    keep_table_in_host_ram()
    if preload:
        host_load_device()
    if table_meta:  # RL1_SHARED_TABLE: the n-gram table comes from rl1_table (built once per launch), not the load
        import rl1_table

        rl1_table.meta_loading()
    stop = threading.Event()
    heartbeat(rec, stop, "load")
    anon = {"peak": 0.0}

    def sample_anon():  # the anonymous peak (VmHWM counts the mmap'd shards' page cache too): the preload guard's need
        while not stop.wait(2.0):
            anon["peak"] = max(anon["peak"], rss_split().get("RssAnon", 0.0))

    threading.Thread(target=sample_anon, daemon=True).start()
    began = time.time()
    phases = {"preload": preload, "table_meta": bool(table_meta)}
    if PRELOAD.get("room"):
        phases["preload_room"] = {k: v for k, v in PRELOAD["room"].items() if k != "ok"}
        if not preload:  # asked, no room: the direct load (H2's path)
            rec.event("preload_skipped", **PRELOAD["room"])
    try:
        model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)
        phases["host_load_s" if preload else "load_s"] = round(time.time() - began, 1)
        if os.environ.get("RL1_LOAD_DIGEST") == "1":  # exactness: a preloaded run's digest equals a direct load's
            phases["digest"] = load_digest(model)
        if preload:
            if before_gate is not None:  # the caller's host-side steps (the trainer's table scale)
                at = time.time()
                phases["before_gate"] = before_gate(model)
                phases["before_gate_s"] = round(time.time() - at, 1)
            phases["cuda_before_gate"] = bool(torch.cuda.is_initialized())  # a step touched the card early
            phases["host_gib"] = host_gib()
            phases["rss_gib"] = {**rss_split(), "anon_peak": anon["peak"]}  # anonymous vs file-backed (page cache)
            print(json.dumps({"event": "rl1_host_preloaded", **{k: v for k, v in phases.items() if k != "before_gate"}}),
                  flush=True)
            phases["gate_wait_s"] = wait_gate(os.environ["RL1_GPU_GATE"])
            phases["shim"] = drop_preload_shims()
            # this rank's card current before anything touches CUDA (rl1_dist deferred its set_device; an unset rank
            # would open a context on card 0)
            torch.cuda.set_device(CARD_INDEX)
            moved_at = time.time()
            phases["moved_gib"] = round(to_card(model, PRELOAD_DEVICE or f"cuda:{CARD_INDEX}") / 2 ** 30, 2)
            torch.cuda.synchronize()
            phases["to_card_s"] = round(time.time() - moved_at, 1)
            phases["release_host"] = release_host()
            reset_peaks()
    finally:
        stop.set()
    torch.cuda.synchronize()
    groups, devices, trainable, total = {}, {}, 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        devices[str(param.device)] = devices.get(str(param.device), 0) + param.numel()
        if param.requires_grad:
            trainable += param.numel()
            groups.setdefault(group_of(name), {"tensors": 0, "params": 0})
            groups[group_of(name)]["tensors"] += 1
            groups[group_of(name)]["params"] += param.numel()
    table = [str(p.device) for n, p in model.named_parameters() if n.endswith("ngram_embedding.weight")]
    experts = next(((n, type(m).__name__) for n, m in model.named_modules() if n.endswith("mlp.experts")), None)
    kinds = {}
    for name, module in model.named_modules():  # which linear layers bitsandbytes holds in 4 bits
        if isinstance(module, torch.nn.Linear) and "lora_" not in name:
            kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
    expert_casts = cast_expert_inputs(model)
    early = phases.pop("before_gate", None)  # the caller's own objects (tensors, hooks): back to it, not to the log
    row = rec.event("loaded", seconds=round(time.time() - began, 1), phases=phases, linear_layers=kinds,
                    expert_casts=expert_casts,
                    gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                    gpu_reserved_gib=round(torch.cuda.memory_reserved() / 2 ** 30, 2), host_peak_gib=host_gib(),
                    params_by_device=devices, params_total=total, trainable=trainable, adapter_groups=groups,
                    ngram_table_on=table, experts_module=experts,
                    peft=None if peft_config is None else {"r": peft_config.r, "alpha": peft_config.lora_alpha})
    if early is not None:
        row["phases"] = {**phases, "before_gate": early}
    return model, tokenizer, row


def cast_expert_inputs(model):
    """BF16 teacher (Stage A smoke, 20 Oct): a float32 residual stream reaches the routed experts and SonicMoE's BF16
    grouped GEMM refuses it ("Type must match: Float32 != BFloat16"); NVFP4 experts cast inside their kernel. A
    pre-hook casts the experts' float32 hidden states to BF16 when the expert weights are BF16 (no packed 4-bit or
    FP8 tensor in the module); routing weights and indices pass unchanged. Returns the number of hooked modules.
    Live in every RL run to 26 Oct 16:05 (48 on Flash); d4727e4a7 dropped it by mistake (T1c-v4: |dlogp| 0.023 vs
    T1a's refs), restored 26 Oct."""
    import torch

    packed = {torch.uint8, torch.int8, getattr(torch, "float8_e4m3fn", torch.uint8)}

    def pre(module, args, kwargs):
        if args and torch.is_tensor(args[0]) and args[0].dtype == torch.float32:
            args = (args[0].to(torch.bfloat16), *args[1:])
        hidden = kwargs.get("hidden_states")
        if torch.is_tensor(hidden) and hidden.dtype == torch.float32:
            kwargs = {**kwargs, "hidden_states": hidden.to(torch.bfloat16)}
        return args, kwargs

    hooked = 0
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not name.endswith("mlp.experts"):
            continue
        dtypes = {p.dtype for p in module.parameters()}
        if torch.bfloat16 in dtypes and not dtypes & packed:
            # the MoE block too: the kernels plugin's SonicMoE forward may read the expert weights without calling
            # the experts module
            for target in (module, modules.get(name.rsplit(".", 1)[0])):
                if target is not None:
                    target.register_forward_pre_hook(pre, with_kwargs=True)
            hooked += 1
    return hooked


def read_only(cache):
    """Make the cache's update calls pure for the window pass: each returns what the layer needs (the prefix state
    joined with the new states) and stores nothing. A gradient-checkpointed layer runs twice, and the stock updates
    append or overwrite in place, so the second run would see its own first run."""
    import torch

    layers = cache.layers

    def rows(state, like):  # a batch-1 prefix state widened to the new states' rows (batched tails)
        return state.expand(like.shape[0], *state.shape[1:]) if state.shape[0] != like.shape[0] else state

    def update(key_states, value_states, layer_idx, *args, **kwargs):
        layer = layers[layer_idx]
        return (torch.cat([rows(layer.keys, key_states), key_states], dim=-2),
                torch.cat([rows(layer.values, value_states), value_states], dim=-2))

    def update_indexer(states, layer_idx):
        return torch.cat([rows(layers[layer_idx].indexer_keys, states), states], dim=1)

    def update_conv_state(conv_states, layer_idx, state_idx=0, **kwargs):
        layer = layers[layer_idx]
        if not layer.has_previous_state[state_idx]:
            raise RuntimeError(f"layer {layer_idx} has no prefix state {state_idx}")
        return torch.cat([rows(layer.conv_states[state_idx], conv_states), conv_states], dim=-1)

    def update_recurrent_state(states, layer_idx, state_idx=0, **kwargs):
        return states

    cache.update, cache.update_indexer = update, update_indexer
    cache.update_conv_state, cache.update_recurrent_state = update_conv_state, update_recurrent_state
    return cache


def served_messages(messages):
    """Chat messages as the served template gets them from vLLM: tool arguments parsed, ``reasoning`` under the
    template's name, images as placeholders (returned apart, in order)."""
    import base64
    import io

    from PIL import Image

    out, images = [], []
    for message in messages:
        message = {key: value for key, value in message.items() if value is not None}
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part.get("type") == "image_url":
                    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    part = {"type": "image"}
                parts.append(part)
            message["content"] = parts
        if message.get("tool_calls"):
            message["tool_calls"] = [{**call, "function": {**call["function"], "arguments": json.loads(
                call["function"]["arguments"])}} for call in message["tool_calls"]]
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        out.append(message)
    return out, images


def encode_request(processor, capture):
    """A captured request (``arena.level_bench`` fork capture: the served payload and the recorded reply) as one
    training example: the prompt exactly as the server tokenized it, then the reply; the loss is on the reply only
    (through its end-of-turn token). ``checks`` compares the counts with the server's own."""
    import torch

    payload, tokenizer = capture["payload"], processor.tokenizer
    settings = dict(tools=payload.get("tools"), tokenize=False, **(payload.get("chat_template_kwargs") or {}))
    messages, images = served_messages(payload["messages"])
    reply, _ = served_messages([{**capture["reply"], "role": "assistant"}])
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **settings)
    whole = tokenizer.apply_chat_template(messages + reply, add_generation_prompt=False, **settings)
    end = "<|im_end|>"
    if not whole.startswith(prompt) or not whole.rstrip("\n").endswith(end):
        raise ValueError("the reply does not render as a continuation of the served prompt")
    whole = whole[: whole.rindex(end) + len(end)]

    def encode(text):
        return processor(text=[text], images=images or None, return_tensors="pt")

    head, inputs = encode(prompt), encode(whole)
    count, ids = head["input_ids"].shape[1], inputs["input_ids"]
    prefix_kept = bool(torch.equal(ids[:, :count], head["input_ids"]))
    exact = {}
    if reply_ids_on() and capture.get("records") is not None:
        inputs, count, exact = served_ids(processor, capture, inputs, count)
        ids = inputs["input_ids"]
    labels = ids.clone()
    labels[:, :count] = -100
    usage = capture.get("usage") or {}
    checks = {"request": capture.get("request"), "prompt_tokens": count, "served_prompt_tokens": capture["prompt_tokens"],
              "reply_tokens": ids.shape[1] - count, "served_reply_tokens": usage.get("completion_tokens"),
              "images": len(images), "image_tokens": int((ids == processor.image_token_id).sum()),
              "prefix_kept": prefix_kept, **exact}
    return {**inputs, "labels": labels}, checks


def reply_ids_on():
    """RL1_REPLY_IDS=1 (29 Oct): windows carry the token ids the play server saw (its reply-id cache replayed from the
    rollout), not a canonical re-tokenization; the served prompt digest must match (``served_ids``)."""
    return os.environ.get("RL1_REPLY_IDS") == "1"


def served_ids(processor, capture, inputs, count):
    """``(inputs, prompt length, checks)`` with the prompt's earlier replies as the server re-sent them
    (``rl_rollouts.substitute`` over the capture's ``records``) and the reply as sampled (its ``rollout.tokens``,
    when the same text). ``checks``: ``prompt_same`` (the served prompt digest matches; None without one), ``spans``
    (earlier replies replaced), ``reply_sampled`` (the reply's own ids replaced)."""
    import torch

    try:
        import rl_rollouts as rr
    except ImportError:
        from posttrain import rl_rollouts as rr
    tok = processor.tokenizer
    ids = inputs["input_ids"][0].tolist()
    image = processor.image_token_id
    images, i = [], 0
    while i < len(ids):  # image token runs, with the vision start / end marks around them
        if ids[i] == image:
            j = i
            while j < len(ids) and ids[j] == image:
                j += 1
            images.append((i - 1, j + 1))
            i = j
        else:
            i += 1
    prompt, splices = rr.substitute(ids[:count], capture["records"], tok, images)
    reply = ids[count:]
    sampled = [int(t) for t in ((capture.get("rollout") or {}).get("tokens") or [])]
    reply_sampled = bool(sampled and reply and sampled != reply and sampled[-1] == reply[-1]
                         and rr.same_text(tok, reply, sampled))
    if reply_sampled:
        splices = splices + [(count, len(ids), sampled)]
    digest = (capture.get("rollout") or {}).get("prompt_digest")
    checks = {"prompt_same": None if not digest else rr.ids_digest(prompt) == digest, "spans": len(splices) - reply_sampled,
              "reply_sampled": reply_sampled}
    if not splices:
        return inputs, count, checks
    total = len(ids)
    out = {}
    for key, value in inputs.items():
        if not (isinstance(value, torch.Tensor) and value.dim() == 2 and value.shape[0] == 1 and value.shape[1] == total):
            out[key] = value
            continue
        row, pieces, cursor = value[0], [], 0
        for start, stop, body in splices:
            pieces.append(row[cursor:start])
            fill = (torch.tensor(body, dtype=row.dtype) if key == "input_ids" else
                    torch.full((len(body),), int(key == "attention_mask"), dtype=row.dtype))
            pieces.append(fill)
            cursor = stop
        pieces.append(row[cursor:])
        out[key] = torch.cat(pieces).unsqueeze(0)
    return type(inputs)(out) if not isinstance(inputs, dict) else out, len(prompt), checks


def image_features(model, inputs, host=None):
    """The window's image features from the frozen vision tower (``positions_and_images``; MFU-1 vision_ahead runs
    the same call in the trainer's feed thread). U2 (27 Oct): with ``host`` (the window's host tensors,
    ``rl1_encode.to_device``) the tower gets its patch positions and block lengths from the host grid (its blocks'
    split sizes need no card read: ~80 syncs a window before) and the features come back unsplit (no split-size read)."""
    import torch

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
    grid = None if host is None else host.get("image_grid_thw")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if grid is None:
            return torch.cat(core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                     return_dict=True).pooler_output, dim=0)
        from transformers import vision_utils
        from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

        visual, pixels = core.visual, inputs["pixel_values"]
        extra = {"position_ids": vision_utils.get_vision_position_ids(grid, visual.spatial_merge_size).to(pixels.device)}
        if not modeling.is_flash_attention_requested(visual.config):  # the per-block split reads cu_seqlens' lengths
            extra["cu_seqlens"] = vision_utils.get_vision_cu_seqlens(grid)
        return visual(pixels.type(visual.dtype), grid_thw=inputs["image_grid_thw"], **extra).pooler_output


def positions_and_images(model, inputs, features=None, marks=None, positions=None, host=None):
    """What a pass over a slice of the sequence needs besides its tokens: M-RoPE positions of the whole sequence
    (the model's own incremental rule only knows the images of its first call) and the image features, computed once
    (the vision tower is frozen). ``piece(start, stop)`` gives the keyword arguments of that slice; an image may
    straddle slices. MFU-1: ``features`` computed earlier (vision_ahead) are used as given; ``marks`` gets the
    seconds of the vision tower and of the position rule. U2 (27 Oct): ``positions`` computed earlier (host M-RoPE,
    ``rl1_encode.host_mrope``) are used as given; with ``host`` (the window's host tensors) the position rule runs on
    the host copies and the per-slice image counts are host reads (~50 card syncs a window before)."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
    ids = inputs["input_ids"]
    if inputs.get("pixel_values") is None:
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
        return lambda start, stop: {"position_ids": positions[:, :, start:stop]}
    host = host if host and all(k in host for k in ("input_ids", "mm_token_type_ids", "image_grid_thw")) else None
    began = time.time()
    if features is None:
        features = image_features(model, inputs, host)
        if marks is not None and ids.is_cuda:
            torch.cuda.synchronize()
    if marks is not None:
        marks["vision"] = round(time.time() - began, 4)
        began = time.time()
    if positions is None and host is not None:
        positions, _ = core.get_rope_index(host["input_ids"], mm_token_type_ids=host["mm_token_type_ids"],
                                           image_grid_thw=host["image_grid_thw"])
    if positions is None:
        positions, _ = core.get_rope_index(ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                           image_grid_thw=inputs["image_grid_thw"])
    if ids.is_cuda and positions.device.type == "cpu":  # pinned: a pageable copy waits for the card's queue
        positions = positions.pin_memory().to(ids.device, non_blocking=True)
    positions = positions.to(ids.device)
    if marks is not None:
        marks["rope"] = round(time.time() - began, 4)
    image_ids = ids if host is None else host["input_ids"]
    before = torch.nn.functional.pad((image_ids[0] == core.config.image_token_id).cumsum(0), (1, 0))

    def piece(start, stop):
        extra = {"position_ids": positions[:, :, start:stop]}
        first, last = int(before[start]), int(before[stop])
        if last > first:
            extra["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features[first:last],))}
        return extra

    return piece


ADAPTER = {"rank": 16, "alpha": 32}


def example_maker(processor, train, *, tail, phases, device=f"cuda:{CARD_INDEX}"):
    """``(encoded, None)``: ``encoded(capture)`` one served request as tensors on ``device`` (None: left on the
    host), or None when it is not the served request (prompt counts within 8 tokens) or does not fit (never
    truncated). ``train`` / ``phases``: unused (the old example builder's arguments)."""

    def encoded(capture, pair=False):
        inputs, checks = encode_request(processor, capture)
        total = inputs["input_ids"].shape[1]
        # RL1_REPLY_IDS: the served prompt digest decides per reply (rl_job.attempt_example leaves a reply whose
        # prompt is not the served one out of the loss); without it, the old count tolerance
        if checks.get("prompt_same") is None and abs(checks["prompt_tokens"] - checks["served_prompt_tokens"]) > 8:
            return None
        if total > 131072 or (not pair and checks["reply_tokens"] + 2 > tail):
            return None
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        return (inputs, checks["prompt_tokens"]) if pair else inputs

    return encoded, None


def apply_table_scale(rec, model, model_dir):
    """The served NVFP4 checkpoint stores the n-gram table in FP8 with a ``weight_scale`` the stock loader drops
    (its load report lists the key as unexpected; the serving recipe patches its own loader for it). Without the
    scale the trainer's base scores at chance (span NLL 11.9, ``sft-qlora-nvfp4-b200-v6``)."""
    import torch
    from safetensors import safe_open

    weight_map = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    done = []
    for key, shard in weight_map.items():
        if not key.endswith("ngram_embedding.weight_scale"):
            continue
        with safe_open(str(Path(model_dir) / shard), framework="pt") as file:
            scale = file.get_tensor(key).float()
        name = key[:-len("_scale")]
        param = next(p for n, p in model.named_parameters() if n.endswith(name))
        stored = sorted(k for k in weight_map if k.startswith(name))[:3]  # the table is stored in pieces
        before = float(param.data[:65536].float().abs().mean())
        step = 1 << 22
        with torch.no_grad():
            for start in range(0, param.shape[0], step):  # in place, in pieces: the table is 51B values
                piece = param.data[start:start + step]
                rows = scale[start:start + step] if scale.dim() and scale.shape[0] == param.shape[0] else scale
                piece.copy_((piece.float() * rows).to(piece.dtype))
        done.append(rec.event("table_scale", key=key, scale_shape=list(scale.shape), stored_keys=stored,
                              scale_mean=float(scale.mean()), table_dtype=str(param.dtype), device=str(param.device),
                              abs_mean_before=round(before, 6),
                              abs_mean_after=round(float(param.data[:65536].float().abs().mean()), 6)))
    return done


