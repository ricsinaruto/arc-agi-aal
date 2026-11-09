"""Launch Pennyroyal's Flash-Next FR-Spec recipe as an explicit ``sglang serve`` argv.

``configs/pennyroyal/serve-flash-next-frspec.sh`` (jpezzulli/sglang-rtxpro6000, tag
pennyroyal-v2.5.1) is a bash launcher that needs the image's Fedora userland, git and the
NIXL host cache. On Kaggle the runtime is the image's ``/opt`` tree extracted next to
Ubuntu, so this module reproduces the script's environment exports and its ``launch_args``
in order, drops the hierarchical-cache block (NIXL is not usable there), binds to
127.0.0.1 and points the compilers at the host's gcc. Everything dropped or changed is
returned by :func:`recipe_diff` and written into every server identity record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .profile import ServerProfile

RECIPE_SCRIPT = "configs/pennyroyal/serve-flash-next-frspec.sh"
RECIPE_IMAGE = "ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.1"
TEMPLATE_RELATIVE = "configs/pennyroyal/templates/froggeric-v22.5.jinja"
TOKEN_MAP_RELATIVE = "configs/pennyroyal/frspec/flash-next-64k.pt"
TEMPLATE_SHA256 = "e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
RECIPE_CONTEXT_LENGTH = 524_288
TARGET_OVERRIDES = (
    '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],'
    '"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,'
    '"original_max_position_embeddings":262144}}}'
)
DEFAULT_CHAT_TEMPLATE_KWARGS = (
    '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
)
DROPPED_RECIPE_FLAGS = [
    "--enable-hierarchical-cache",
    "--hicache-size 32",
    "--hicache-host-memory-mode cache",
    "--hicache-write-policy write_through",
    "--hicache-io-backend kernel",
    "--hicache-mem-layout page_first",
    "--hicache-storage-backend nixl",
    "--hicache-storage-prefetch-policy timeout",
    "--hicache-storage-backend-extra-config @nixl-posix-frspec.toml",
]
DROPPED_RECIPE_ENV = ["SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR", "NIXL_STORAGE_BASE"]
RECIPE_CC, RECIPE_CXX, RECIPE_CUDA_HOME = "/usr/bin/gcc-15", "/usr/bin/g++-15", "/usr/local/cuda-13.3"
# The first smoke (19 September) loaded the target's 206 shards in 1302 s from Kaggle's NFS
# (4 s/shard cold, 38 s/shard at the end) and then re-read them all for the NEXTN draft,
# so 1500 s was not enough; runtime.prefetch_model warms the page cache first.
READY_TIMEOUT = 2700
STOP_GRACE = 120

STARTUP_PATTERNS = {
    "max_total_num_tokens": r"max_total_num_tokens=(\d+)",
    "chunked_prefill_size": r"chunked_prefill_size=(\d+)",
    "max_prefill_tokens": r"max_prefill_tokens=(\d+)",
    "max_running_requests": r"max_running_requests=(\d+)",
    "context_len": r"context_len=(\d+)",
    "available_gpu_mem_gb": r"available_gpu_mem=([\d.]+) GB",
    "max_mamba_cache_size": r"max_mamba_cache_size:\s*(\d+)",
}
REQ_STATS = re.compile(
    r"ReqTimeStats\(rid=(?P<rid>[^,]+), input_len=(?P<input_len>\d+), "
    r"cached_input_len=(?P<cached_input_len>\d+), output_len=(?P<output_len>\d+), "
    r"attempts=(?P<attempts>\d+)(?:, type=(?P<type>[^)]+))?\):(?P<rest>.*)"
)
REQ_FIELD = re.compile(r"(\w+)=([-\d.eE+]+)(ms|us|µs|ns|s)?(?=[,\s]|$)")


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def recipe_argv(profile: ServerProfile, *, model_dir, repo_root, sglang_exe) -> list[str]:
    """The script's ``launch_args`` in order, without the hierarchical-cache block."""
    repo_root, model_dir = Path(repo_root), Path(model_dir)
    argv = [
        str(sglang_exe), "serve",
        "--model-path", str(model_dir),
        "--load-format", "safetensors",
        "--served-model-name", profile.served_model_name,
        "--host", profile.host, "--port", str(profile.port), "--tp", "1",
        "--dtype", "bfloat16", "--quantization", "modelopt_fp4", "--kv-cache-dtype", "fp8_e4m3",
        "--mem-fraction-static", "0.981",
        "--max-total-tokens", str(profile.max_total_tokens), "--warmups=structured_output",
        "--context-length", str(profile.context_length),
    ]
    if profile.context_length == RECIPE_CONTEXT_LENGTH:
        argv += ["--json-model-override-args", TARGET_OVERRIDES]
    argv += [
        "--page-size", str(profile.page_size),
        "--max-running-requests", str(profile.admitted), "--sleep-on-idle",
        "--chunked-prefill-size", str(profile.chunked_prefill_size),
        "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-ssm-dtype", "bfloat16",
        "--max-mamba-cache-size", str(profile.mamba_slots), "--gdn-mtp-cache-mode", "none",
        "--linear-attn-decode-backend", "flashinfer", "--linear-attn-prefill-backend", "flashinfer",
        "--mamba-track-interval", "64",
        "--ple-offload-embedding", "--trust-remote-code",
        "--chat-template", str(repo_root / TEMPLATE_RELATIVE),
        "--image-processor-backend", "pil" if profile.mm_preprocess_device == "cpu" else "torchvision",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--enable-request-time-stats-logging", "--enable-metrics",
        "--default-chat-template-kwargs", DEFAULT_CHAT_TEMPLATE_KWARGS,
        "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
        "--speculative-draft-model-quantization", "unquant",
        "--speculative-token-map", str(repo_root / TOKEN_MAP_RELATIVE),
        "--watchdog-timeout", "1800",
    ]
    if profile.weight_loader_disable_mmap:
        argv.append("--weight-loader-disable-mmap")
    if profile.weight_loader_threads is not None:
        argv += ["--model-loader-extra-config", json.dumps({
            "enable_multithread_load": True, "num_threads": profile.weight_loader_threads,
        }, separators=(",", ":"))]
    return argv


def recipe_env(
    profile: ServerProfile,
    *,
    cache_base,
    cuda_home,
    repo_root,
    cc="/usr/bin/gcc",
    cxx="/usr/bin/g++",
    base=None,
    scratch=None,
) -> dict[str, str]:
    """The script's exports (plus the image's) over ``base``; cache directories are created."""
    cache, cuda, repo = Path(cache_base), Path(cuda_home), Path(repo_root)
    scratch = Path(scratch) if scratch else cache / "tmp"
    base = dict(os.environ if base is None else base)
    for name in DROPPED_RECIPE_ENV:
        base.pop(name, None)
    venv_bin = repo / ".venv/bin"
    for sub in ("huggingface", "torch", "torchinductor", "triton", "cuda", "flashinfer", "sglang/jit"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    library_path = ":".join(
        p for p in (str(cuda / "lib64"), "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu",
                    base.get("LD_LIBRARY_PATH", "")) if p
    )
    env = {
        **base,
        "PATH": ":".join([str(venv_bin), str(cuda / "bin"), base.get("PATH", "/usr/bin:/bin")]),
        "LD_LIBRARY_PATH": library_path,
        "PYTHONPATH": str(repo / "python"),
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "SGLANG_MM_PREPROCESS_DEVICE": profile.mm_preprocess_device,
        "SGLANG_FORWARD_UNKNOWN_TOOLS": "true",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": base.get("CUDA_VISIBLE_DEVICES", "0"),
        "CUDA_HOME": str(cuda),
        "CUDACXX": str(cuda / "bin/nvcc"),
        "CC": cc, "CXX": cxx, "CUDAHOSTCXX": cxx, "NVCC_CCBIN": cxx,
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "PENNY_BUILD_JOBS": "4", "MAX_JOBS": "4", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CARGO_BUILD_JOBS": "4", "FLASHINFER_NINJA_JOBS": "4", "FLASHINFER_NVCC_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "SGLANG_CACHE_DIR": str(cache / "sglang"), "SGLANG_JIT_CACHE_DIR": str(cache / "sglang/jit"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_NUMA_BIND_V2": "false", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "SGLANG_SM120_ONLINE_MXFP8": "true" if profile.online_fp8 else "false",
        "PENNY_PLE_BACKEND": profile.ple_backend,
        # The image's own environment and offline guards.
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(scratch),
    }
    return env


def recipe_diff(env: dict[str, str], argv: list[str]) -> dict:
    """What differs from ``serve-flash-next-frspec.sh`` as shipped in the image."""
    return {
        "script": RECIPE_SCRIPT,
        "image": RECIPE_IMAGE,
        "dropped_flags": list(DROPPED_RECIPE_FLAGS),
        "dropped_env": list(DROPPED_RECIPE_ENV),
        "weight_loading": {
            "disable_mmap": "--weight-loader-disable-mmap" in argv,
            "extra_config": (json.loads(argv[argv.index("--model-loader-extra-config") + 1])
                             if "--model-loader-extra-config" in argv else {}),
        },
        "changed": {
            "--host": ["0.0.0.0", argv[argv.index("--host") + 1]],
            "CC": [RECIPE_CC, env["CC"]],
            "CXX": [RECIPE_CXX, env["CXX"]],
            "CUDA_HOME": [RECIPE_CUDA_HOME + " (image)", env["CUDA_HOME"]],
            "launcher": ["bash recipe + derive_namespace.py + startup-summary.sh", "sglang_bench.launch"],
        },
        "kept_pins": {
            "template_sha256": TEMPLATE_SHA256,
            "token_map_sha256": TOKEN_MAP_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "json_model_override_args": TARGET_OVERRIDES if TARGET_OVERRIDES in argv else None,
        },
    }


def gpu_snapshot() -> dict:
    """Memory and utilisation of the GPU from nvidia-smi (errors recorded, never raised)."""
    from arc_infra.metrics import GPU_FIELDS, parse_gpu_rows

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return {"gpus": parse_gpu_rows(out)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def port_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) != 0


def preflight(profile: ServerProfile, *, model_dir, repo_root) -> dict:
    model_dir, repo_root = Path(model_dir), Path(repo_root)
    problems = []
    for name in ("config.json", "model.safetensors.index.json", "chat_template.jinja", "tokenizer.json"):
        if not (model_dir / name).is_file():
            problems.append(f"model file missing: {name}")
    pins = {}
    for label, path, expected in (
        ("tokenizer", model_dir / "tokenizer.json", TOKENIZER_SHA256),
        ("template", repo_root / TEMPLATE_RELATIVE, TEMPLATE_SHA256),
        ("token_map", repo_root / TOKEN_MAP_RELATIVE, TOKEN_MAP_SHA256),
    ):
        if path.is_file():
            pins[label] = sha256_file(path)
            if pins[label] != expected:
                problems.append(f"{label} sha256 {pins[label]} != recipe pin {expected}")
        else:
            problems.append(f"missing {label}: {path}")
    if profile.max_total_tokens % profile.page_size:
        problems.append("max_total_tokens not page aligned")
    if not port_free(profile.host, profile.port):
        problems.append(f"port {profile.port} is in use")
    record = {"problems": problems, "pins": pins, "gpu_before": gpu_snapshot()}
    if problems:
        raise RuntimeError("Preflight failed: " + "; ".join(problems))
    return record


@dataclass
class Server:
    process: subprocess.Popen
    argv: list[str]
    log_path: Path
    identity_path: Path
    started_epoch: float
    profile: ServerProfile

    @property
    def base_url(self) -> str:
        return self.profile.base_url


def start(profile: ServerProfile, *, working, env, argv, preflight_record=None) -> Server:
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    log_path = working / "sglang-server.log"
    identity = {
        "profile": profile.to_dict(),
        "argv": argv,
        "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        "env": {k: v for k, v in env.items() if k.startswith(("SGLANG", "PENNY", "CUDA", "TORCH",
                 "PYTORCH", "HF_", "OMP", "MKL", "CC", "CXX", "NVCC", "LD_LIBRARY_PATH", "PATH",
                 "PYTHONPATH", "TMPDIR", "FLASHINFER", "TRITON", "XDG", "NUMPY", "TOKENIZERS",
                 "MAX_JOBS", "CMAKE", "CARGO"))},
        "recipe_diff": recipe_diff(env, argv),
        "preflight": preflight_record,
        "started_epoch": time.time(),
    }
    identity_path = working / "server-identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    log = log_path.open("ab")
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=str(working))
    log.close()
    identity["pid"] = process.pid
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return Server(process, argv, log_path, identity_path, identity["started_epoch"], profile)


def tail(path: Path, lines: int = 300) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_ready(server: Server, *, timeout=READY_TIMEOUT, poll=5, log=print) -> dict:
    """Poll /health, then /v1/models; abort if the process dies."""
    import httpx
    from .loading_metrics import append_sample

    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_report = started
    client = httpx.Client(trust_env=False, timeout=10)
    healthy = False
    resource_log = server.log_path.with_name('loading-resources.jsonl')
    try:
        while time.monotonic() < deadline:
            append_sample(server.process.pid, resource_log)
            code = server.process.poll()
            if code is not None:
                raise RuntimeError(f"sglang exited {code} before ready:\n{tail(server.log_path)}")
            try:
                if not healthy:
                    healthy = client.get(server.base_url.removesuffix("/v1") + "/health").status_code == 200
                if healthy:
                    models = client.get(server.base_url + "/models").json()
                    ids = [m.get("id") for m in models.get("data", [])]
                    if server.profile.served_model_name in ids:
                        record = {"ready_seconds": round(time.monotonic() - started, 1), "models": ids}
                        record.update(parse_startup(Path(server.log_path).read_text(errors="replace")))
                        return record
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() - last_report >= 60:
                last_report = time.monotonic()
                log(f"[sglang] waiting {int(last_report - started)} s; log tail: {tail(server.log_path, 3)}")
            time.sleep(poll)
    finally:
        client.close()
        append_sample(server.process.pid, resource_log)
    raise TimeoutError(f"sglang not ready after {timeout} s:\n{tail(server.log_path)}")


def parse_startup(text: str) -> dict:
    record = {}
    for key, pattern in STARTUP_PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            value = found[-1]
            record[key] = float(value) if "." in value else int(value)
    record["weights_loaded"] = "Load weight end" in text
    record["ready_banner"] = "fired up and ready" in text
    record["jit_compile_lines"] = sum(1 for line in text.splitlines() if re.search(r"\b(nvcc|ninja)\b", line))
    return record


def clamp_check(profile: ServerProfile, startup: dict) -> dict:
    fitted = startup.get("max_total_num_tokens")
    return {
        "requested_tokens": profile.max_total_tokens,
        "fitted_tokens": fitted,
        "clamped": fitted is not None and fitted != profile.max_total_tokens,
        "requested_mamba": profile.mamba_slots,
        "fitted_mamba": startup.get("max_mamba_cache_size"),
    }


def loop_payload(model: str, messages: list[dict], *, tools=True, max_tokens=None) -> dict:
    """A request shaped like the public loop's (``inference.utils.openai_compat``)."""
    from inference.utils.openai_compat import build_chat_payload

    tool = [{
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run Python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }]
    return build_chat_payload(
        provider="vllm", model=model, messages=messages, max_tokens=max_tokens, temperature=0.6,
        top_p=0.95, top_k=20, thinking=True, tools=tool if tools else None,
        tool_choice="auto" if tools else None, seed=None,
    )


def post_chat(base_url: str, payload: dict, *, timeout=600) -> dict:
    import httpx

    started = time.monotonic()
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        response = client.post(base_url + "/chat/completions", json=payload)
    record = {"status": response.status_code, "seconds": round(time.monotonic() - started, 2)}
    try:
        body = response.json()
    except ValueError:
        body = {"text": response.text[:2000]}
    if response.status_code == 200:
        message = (body.get("choices") or [{}])[0].get("message") or {}
        record.update(
            usage=body.get("usage"),
            finish_reason=(body.get("choices") or [{}])[0].get("finish_reason"),
            has_reasoning=bool(message.get("reasoning_content")),
            tool_calls=len(message.get("tool_calls") or []),
            content_chars=len(message.get("content") or ""),
        )
    else:
        record["body"] = json.dumps(body)[:2000]
    return record


def warmup(server: Server, *, probes=False) -> dict:
    """One loop-shaped completion; optionally the two over-long-prompt probes."""
    model = server.profile.served_model_name
    messages = [{"role": "user", "content": "Reply with the single word ready."}]
    record = {"chat": post_chat(server.base_url, loop_payload(model, messages))}
    if probes:
        # The loop's overflow detector matches "maximum context length"; SGLang emits it only
        # when max_tokens pushes past the window, and a different text for an over-long input.
        record["probe_max_tokens"] = post_chat(
            server.base_url, loop_payload(model, messages, tools=False, max_tokens=600_000), timeout=120
        )
        long = [{"role": "user", "content": "a " * (server.profile.context_length + 20_000)}]
        record["probe_long_input"] = post_chat(server.base_url, loop_payload(model, long, tools=False), timeout=300)
        for key in ("probe_max_tokens", "probe_long_input"):
            body = record[key].get("body", "")
            record[key]["matches_loop_detector"] = "maximum context length" in body.lower()
    return record


def stop(server: Server, *, grace=STOP_GRACE) -> dict:
    """SIGTERM the process group, then SIGKILL; wait for the port and GPU memory to free."""
    record = {"pid": server.process.pid}
    started = time.monotonic()
    if server.process.poll() is None:
        try:
            os.killpg(server.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.process.wait(timeout=grace)
            record["stopped_by"] = "SIGTERM"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.process.wait()
            record["stopped_by"] = "SIGKILL"
    else:
        record["stopped_by"] = "already exited"
    record["returncode"] = server.process.returncode
    for _ in range(60):
        if port_free(server.profile.host, server.profile.port):
            break
        time.sleep(1)
    record["port_free"] = port_free(server.profile.host, server.profile.port)
    record["gpu_after"] = gpu_snapshot()
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def request_time_stats(text: str) -> list[dict]:
    rows = []
    for match in REQ_STATS.finditer(text):
        row = {k: (int(v) if v is not None and v.isdigit() else v) for k, v in match.groupdict().items() if k != "rest"}
        for key, value, unit in REQ_FIELD.findall(match.group("rest")):
            try:
                row[key] = float(value) * {"": 1, "s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}[unit]
            except ValueError:
                pass
        rows.append(row)
    return rows


def cold_vs_warm(rows: list[dict]) -> dict:
    """First request versus the rest: cached share of the prompt and initial prefill time."""
    def summary(subset):
        if not subset:
            return None
        prompt = sum(r.get("input_len", 0) for r in subset)
        cached = sum(r.get("cached_input_len", 0) for r in subset)
        prefill = [r["initial_prefill_elapsed"] for r in subset if "initial_prefill_elapsed" in r]
        return {
            "requests": len(subset),
            "cached_share": round(cached / prompt, 4) if prompt else None,
            "mean_initial_prefill_seconds": round(sum(prefill) / len(prefill), 3) if prefill else None,
            "mean_output_len": round(sum(r.get("output_len", 0) for r in subset) / len(subset), 1),
        }
    return {"first": summary(rows[:1]), "rest": summary(rows[1:]), "all": summary(rows)}
