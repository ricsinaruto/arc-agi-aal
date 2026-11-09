"""MFU-1 harness (heal_kd): one-card proxy cases, GPU lever checks, CPU gates/legs, NCCL bandwidth probe. Rows go through
tools/ops/records.py; usage in (MFU-1 line)."""
import json, sys, time
from pathlib import Path
W = str(Path(__file__).resolve().parent)
sys.path.insert(0, W)
import modal_heal_tiny as m  # noqa: E402

app = m.app

@app.local_entrypoint()
def go(args: str = "", env: str = "", cases: str = "off", save: str = "", gpu: str = "B300", check: str = "",
       cpu_gate: bool = False, inert: bool = False, only: str = "", notiny: bool = False, nccl: str = ""):
    if nccl:  # "<gpus>|case;case" e.g. "2|;NCCL_PROTO=Simple": all-gather bandwidth + transport on B300:<gpus>
        gpus, _, envs = nccl.partition("|")
        r = m.nccl_bw.with_options(gpu=f"B300:{gpus}").remote(envs)
        if save:
            Path(save).write_text(json.dumps(r, indent=1))
        print(json.dumps(r)[:6000])
        return
    if inert:  # CPU: inert-check bars (incl. HEAL_SKIP_INERT_LOSS bit-equality) + stream legs + heal tests
        r = m.inert_cpu.remote(only)
        if save:
            Path(save).write_text(json.dumps(r, indent=1))
        print(json.dumps({"verdict": (r.get("check") or {}).get("verdict"), "skip_inert": r.get("skip_inert"), "align": r.get("align"), "lever1": r.get("lever1"), "kd_rows": r.get("kd_rows"), "step_plan": r.get("step_plan"), "pytest": r.get("pytest", {}).get("tail", "")[-400:]})[:4000])
        return
    if cpu_gate:  # launch gate: CPU import check in the trainer image + heal_kd tiny (CPU)
        print(json.dumps(m.import_check.remote()))
        if notiny:
            return
        t = m.tiny.remote()
        print(json.dumps(t)[:3000])
        return
    if check:
        res = m.gpu_checks.with_options(gpu=gpu).remote(check)
        if save:
            Path(save + ".checks.json").write_text(json.dumps(res, indent=1))
    if not args:
        return
    call = m.proxy.with_options(gpu=gpu).spawn(args, env, cases, 0)
    print(json.dumps({"call_id": call.object_id, "t": time.time()}), flush=True)
    result = call.get()
    if save:
        Path(save).write_text(json.dumps(result, indent=1))
    for e in result["events"]:
        if e.get("event") in ("proxy_case", "proxy_setup"):
            print(json.dumps({k: v for k, v in e.items() if k not in ("windows", "profile")})[:3000])
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=1)[:6000])
