"""GDN + QSA apply-test of a v9 bundle (kern-gdn handoff 29 Oct): copies of the real pinned target files, stack_kernels
(async_h2d + gdn_defer) applied with nvcc stubbed (the .so build runs at server setup on the VM), then serving_kernels
(qsa_fast) on the same copies; checks hooks appended to both GDN files, helper/kernel/lib in vllm/, QSA leaves the GDN
files alone, graphs <= 256 and >= max_num_seqs x 3. Usage: gdn_applytest.py BUNDLE_SUB_DIR"""
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

B = Path(sys.argv[1])
PIN = Path("~/arc-shared/pinned-vllm")
T = Path("<scratch>/v9/gdnapply")
shutil.rmtree(T, ignore_errors=True)
sys.path.insert(0, str(B / "src"))
from arc_infra import serving_kernels_patch as skp  # noqa: E402
from arc_infra import stack_kernels_patch as stk  # noqa: E402

study = json.loads((B / "configs/comparison.json").read_text())
srv = study["server"]
stack_spec, serv_spec = stk.validate(srv["stack_kernels"]), skp.validate(srv["serving_kernels"])
assert stack_spec == {"async_h2d": True, "gdn_defer": True}, stack_spec
gtargets = stk.targets(stack_spec)
qtargets = skp.targets(serv_spec)
for t in gtargets + qtargets:
    dst = T / "site" / t["relative"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PIN / t["relative"], dst)
site = T / "site"
before = {t["relative"]: hashlib.sha256((site / t["relative"]).read_bytes()).hexdigest() for t in gtargets}


def stub(command, **kw):
    Path(command[command.index("-o") + 1]).write_bytes(b"stub-so")
    return types.SimpleNamespace(returncode=0, stderr="")


orig = stk.install_gdn_defer
stk.install_gdn_defer = lambda s, run=None: orig(s, run=stub)
ident = stk.apply(site, T / "out", srv["stack_kernels"])
gdn_files = [t["relative"] for t in gtargets if t is not stk.SHORT_CONV]
print("gdn files", gdn_files)
for rel in gdn_files:
    text = (site / rel).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() != before[rel], rel
    compile(text, rel, "exec")
for f in (stk.GDN_DEFER_HELPER, stk.GDN_DEFER_SOURCE, stk.GDN_DEFER_LIB):
    assert (site / "vllm" / f).exists(), f
assert (site / "vllm" / stk.GDN_DEFER_HELPER).read_bytes() == (B / "src/arc_infra/gdn_defer_runtime.py").read_bytes()
after_gdn = {rel: (site / rel).read_bytes() for rel in gdn_files}
print("stack identity files", sorted(ident["files"]))
overlap = sorted({t["relative"] for t in qtargets} & set(gdn_files))
print("qsa targets touching GDN files:", overlap)
assert not overlap
skp_ident = skp.apply(site, T / "out", srv["serving_kernels"]) if hasattr(skp, "apply") else None
assert all((site / rel).read_bytes() == after_gdn[rel] for rel in gdn_files), "serving_kernels changed a GDN file"
print("qsa applied:", bool(skp_ident), "qsa helper sha", (skp_ident or {}).get("qsa_fast_helper_sha256"))
need = srv["max_num_seqs"] * (1 + int(srv["mtp_tokens"]))
assert need <= srv["max_cudagraph_capture_size"] <= 256, (need, srv["max_cudagraph_capture_size"])
print("graphs", srv["max_cudagraph_capture_size"], ">= seqs x 3 =", need, "<= 256")
print("GDN+QSA APPLY OK")
