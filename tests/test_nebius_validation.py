"""The Nebius backend without credentials: planning, rendering, ledger math, path mapping, dry run, bookkeeping."""

import ast
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nv = load("nebius_validation")
runner = load("nebius_runner")

CONFIG = {"project_id": "project-e00x", "region": "eu-south1", "subnet_id": "vpcsubnet-e00x",
          "service_account_id": "serviceaccount-e00x", "bucket": "arc-bucket", "s3_secret_id": "mbsec-e00x",
          "pricing_policy_id": "pricingpolicy-e00x", "spot_max_price_usd_per_gpu_hour": 1.2}
OVERLAY = "richardcsaky/arc-expert-int3-requant-t37-excl5-v1"
STUDY = {"notebook_seconds": 86100, "wall_seconds": 82200, "trials": [{"name": "duck-transfer", "kind": "duck-transfer"}],
         "model_datasets": [OVERLAY], "server": {"expert_overlay": {"dataset": OVERLAY, "check": "size"}},
         "b200": {"test": "b200-run", "harness": "arena", "gpu": "RTX-PRO-6000",
                  "stop": {"score_target": 95.0, "wall_seconds": 82200, "usd_limit": None}}}


APPROVED = "test approval of a long run"


def launch_spot(*args, **kwargs):
    """A preemptible launch: what the relaunch, resume and controller tests are about (the default is fixed)."""
    return nv.launch(*args, **{"preemptible": True, **kwargs})


def make_bundle(tmp_path, study=STUDY, checkpointing=False):
    bundle = tmp_path / "bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "configs").mkdir(parents=True)
    if checkpointing:  # a bundle built from a branch that has the arena checkpoint code
        (bundle / "src/arena").mkdir(parents=True)
        (bundle / "src/arena/checkpoint.py").write_text("# resumable\n")
    (bundle / "dataset-upload").mkdir()
    (bundle / "dataset-upload/blob.bin").write_bytes(b"x" * 10)
    (bundle / "experiment.ipynb").write_text("{}")
    files = {"configs/comparison.json": json.dumps(study), "bootstrap_arena.py": "print('hi')\n"}
    for name, text in files.items():
        (bundle / name).write_text(text)
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "dependencies_prepared": True,
        "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}}))
    return bundle


def make_ledger(tmp_path, cap=None):
    (tmp_path / "docs").mkdir(exist_ok=True)
    path = tmp_path / "docs/spend.json"
    data = {"cap_usd": 5, "infrastructure_reserve_usd": 2, "runs": [
        {"run_id": "modal-1", "state": "reserved", "charged_bound_usd": 1.0}]}
    path.write_text(json.dumps(data, indent=1) + "\n")
    if cap is not None:
        nv.set_cap(tmp_path, cap, "test cap")
    return path


class FakeCloud(nv.Cloud):
    """Answers the calls a launch makes; records every argv."""

    def __init__(self, config, objects=None, instance_state="RUNNING"):
        super().__init__(config)
        self.objects, self.state, self.deleted = objects or {}, instance_state, []

    version, lacking = "0.12.286", ()  # this side's CLI: its version, and create flags it does not know

    def run(self, argv, *, display=None, check=True, timeout=None):
        done = subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1:] == ["version"] or "--help" in argv:  # local CLI calls, no cloud
            flags = {a for spot in (True, False) for policy in ("p", None) for a in nv.create_command(
                {**self.config, "pricing_policy_id": policy}, "x", {"a": "b"}, "", preemptible=spot) if a[:2] == "--"}
            done.stdout = self.version + "\n" if argv[1:] == ["version"] else "".join(
                f"  \x1b[1m{flag}\x1b[0m (string)\n" for flag in sorted(flags - set(self.lacking)))
            return done
        self.calls.append(argv)
        def at(remote):
            return remote.split(":", 1)[1].split("/", 1)[1]

        if argv[0] == "rclone" and argv[1] == "lsf" and "--dirs-only" in argv:
            done.stdout = "\n".join(sorted({k.split("/")[1] + "/" for k in self.objects if k.startswith("launch/")}))
        elif argv[0] == "rclone" and argv[1] == "copyto" and argv[3].endswith(".json"):
            self.objects[at(argv[3])] = json.loads(Path(argv[2]).read_text())  # the bucket keeps JSON objects
        elif argv[0] == "rclone" and argv[1] == "moveto":
            if at(argv[2]) in self.objects:
                self.objects[at(argv[3])] = self.objects.pop(at(argv[2]))
        elif argv[0] == "rclone" and argv[1] == "lsf":
            done.stdout = "\n".join(k.split("/")[-1] for k in self.objects if "assets-manifests/" in k)
        elif argv[0] == "rclone" and argv[1] == "cat":
            key = argv[2].split(":", 1)[1].split("/", 1)[1]
            if key in self.objects:
                done.stdout = json.dumps(self.objects[key])
            else:
                done.returncode = 3
        elif "create" in argv:
            self.created = getattr(self, "created", 0) + 1
            done.stdout = json.dumps({"metadata": {"id": "computeinstance-e00new" if self.created == 1
                                                   else f"computeinstance-e00new{self.created}"}})
        elif "get" in argv:
            done.stdout = json.dumps({"status": {"state": self.state}}) if self.state else ""
            done.returncode = 0 if self.state else 1
        elif "delete" in argv:
            self.deleted.append(argv[argv.index("--id") + 1])
        elif "list" in argv:
            done.stdout = json.dumps({"items": []})
        return done


def manifests(assets):
    return {f"assets-manifests/{nv.asset_slug(p)}.json": {} for p in assets}


# -- planning ----------------------------------------------------------------------------------------------------
def test_b200_plan_matches_modal_launcher():
    plan = nv.plan_run("arena", "b200-run", STUDY, "b2-v71p-24h-v1-nebius-20261007")
    assert plan == {"timeout": 86400, "script": "bootstrap_arena.py", "arguments": ["--trial", "duck-transfer"],
                    "stop_rule": STUDY["b200"]["stop"]}
    smoke = {**STUDY, "b200": {**STUDY["b200"], "test": "b200-smoke"}}
    assert nv.plan_run("arena", "b200-smoke", smoke, "b2-x")["timeout"] == 1740
    with pytest.raises(ValueError, match="b2- run ids"):
        nv.plan_run("arena", "b200-run", STUDY, "v71p-24h")
    with pytest.raises(ValueError, match="RTX PRO 6000"):
        nv.plan_run("arena", "b200-run", {**STUDY, "b200": {**STUDY["b200"], "gpu": "B200"}}, "b2-x")
    with pytest.raises(ValueError, match="arena bundles only"):
        nv.plan_run("sglang", "smoke", STUDY, "x")
    with pytest.raises(ValueError, match="teacher"):
        nv.plan_run("arena", "duck-110", {**STUDY, "teacher": {"model": "x"}}, "x")


def test_other_tests_use_the_modal_tables():
    plan = nv.plan_run("arena", "duck-110", STUDY, "x")
    assert plan["timeout"] == 7200 and plan["arguments"] == ["--trial", "duck-transfer"]
    assert nv.plan_run("arena", "smoke", STUDY, "x")["arguments"] == ["--smoke"]
    with pytest.raises(ValueError, match="Unsupported"):
        nv.plan_run("arena", "no-such-test", STUDY, "x")


def test_required_assets_are_base_plus_overlays():
    assets = nv.required_assets(STUDY)
    assert assets[:4] == list(nv.BASE_ASSETS)
    assert assets[4:] == [f"flash-input/datasets/{OVERLAY}"]
    assert nv.asset_slug(assets[4]) == "flash-input--datasets--richardcsaky--arc-expert-int3-requant-t37-excl5-v1"


def test_attempt_ids():
    assert nv.attempt_run_id("b2-x", 1) == "b2-x" and nv.attempt_run_id("b2-x", 3) == "b2-x-a3"
    assert nv.split_attempt("b2-x-a3") == ("b2-x", 3) and nv.split_attempt("b2-x") == ("b2-x", 1)


# -- config, rates, ledger ---------------------------------------------------------------------------------------
def test_config_checks():
    config = nv.check_config(CONFIG)
    assert config["platform"] == "gpu-rtx6000-a" and config["endpoint"] == "https://storage.eu-south1.nebius.cloud"
    assert nv.check_config({**CONFIG, "region": "us-central1"})["platform"] == "gpu-rtx6000"
    with pytest.raises(ValueError, match="offered in"):
        nv.check_config({**CONFIG, "region": "eu-north1"})
    with pytest.raises(ValueError, match="93 GiB"):
        nv.check_config({**CONFIG, "boot_disk_gib": 700})
    with pytest.raises(ValueError, match="go together"):
        nv.check_config({**CONFIG, "spot_max_price_usd_per_gpu_hour": None})
    # A config without a spot price serves fixed instances; only a preemptible launch needs one.
    fixed_only = nv.check_config({**CONFIG, "pricing_policy_id": None, "spot_max_price_usd_per_gpu_hour": None})
    assert nv.hourly_rate(fixed_only, preemptible=False)["gpu_usd_per_hour"] == 1.80
    with pytest.raises(ValueError, match="pricing policy"):
        nv.hourly_rate(fixed_only, preemptible=True)
    with pytest.raises(ValueError, match="lacks"):
        nv.check_config({k: v for k, v in CONFIG.items() if k != "bucket"})


def test_rate_and_bound_math():
    config = nv.check_config(CONFIG)
    disk = 744 * 0.053 / 730
    rate = nv.hourly_rate(config)
    assert rate["usd_per_hour"] == pytest.approx(1.2 + disk) and "max spot price" in rate["basis"]
    follow = nv.check_config({**CONFIG, "pricing_policy_id": None, "spot_max_price_usd_per_gpu_hour": None,
                              "follow_spot_price": True})
    assert nv.hourly_rate(follow)["gpu_usd_per_hour"] == 1.80  # no max price: bounded by the on-demand price
    assert nv.hourly_rate(config, preemptible=False)["gpu_usd_per_hour"] == 1.80
    with pytest.raises(ValueError, match="at most the on-demand"):
        nv.hourly_rate(nv.check_config({**CONFIG, "spot_max_price_usd_per_gpu_hour": 2.5}))
    seconds = 86400 + 2400 + 300
    assert nv.bound_usd(seconds, rate["usd_per_hour"]) == pytest.approx((seconds + 60) * (1.2 + disk) / 3600 * 1.15)


def test_ledger_needs_its_own_cap_and_keeps_the_modal_books(tmp_path):
    path = make_ledger(tmp_path)
    before = path.read_text()
    entry = {"run_id": "b2-n1", "state": "reserved", "platform": "nebius", "charged_bound_usd": 30.0,
             "rate_usd_per_hour": 1.254}
    with pytest.raises(RuntimeError, match="No Nebius spend cap"):
        nv.reserve(tmp_path, dict(entry))
    assert path.read_text() == before
    with pytest.raises(ValueError, match="authorization"):
        nv.set_cap(tmp_path, 50, " ")
    nv.set_cap(tmp_path, 50, " for Nebius tests")
    assert nv.reserve(tmp_path, dict(entry)) == 30.0
    raw = path.read_text()
    data = json.loads(raw)
    assert raw == json.dumps(data, indent=1) + "\n"  # the Modal launcher's formatting
    assert data["cap_usd"] == 5 and [r["run_id"] for r in data["runs"]] == ["modal-1"]
    assert data["nebius"]["runs"] == [entry] and data["nebius"]["cap_usd"] == 50
    with pytest.raises(ValueError, match="already reserved"):
        nv.reserve(tmp_path, dict(entry))
    with pytest.raises(ValueError, match="already reserved"):
        nv.reserve(tmp_path, {**entry, "run_id": "modal-1"})
    with pytest.raises(RuntimeError, match="Nebius spend cap"):
        nv.reserve(tmp_path, {**entry, "run_id": "b2-n2", "charged_bound_usd": 20.01})
    assert path.read_text() == raw

    # The Modal launcher's reserve still sees only its own runs and cap.
    tree = ast.parse((SCRIPTS / "modal_validation.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reserve")
    ns = {"ROOT": tmp_path, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation.py", "exec"), ns)
    with pytest.raises(RuntimeError, match=r"Spend cap: \.00"):
        ns["reserve"]("modal-2", "arena", 1680)


def test_settle_lowers_the_bound_and_never_raises_it(tmp_path):
    make_ledger(tmp_path, cap=100)
    nv.reserve(tmp_path, {"run_id": "r", "state": "reserved", "charged_bound_usd": 30.0, "rate_usd_per_hour": 1.254})
    entry = nv.settle_entry(tmp_path, "r", "preempted", 3600)
    assert entry["state"] == "preempted" and entry["previous_charged_bound_usd"] == 30.0
    assert entry["charged_bound_usd"] == pytest.approx((3600 + 120) * 1.254 / 3600 * 1.15)
    assert nv.settle_entry(tmp_path, "r", "complete", 10)["state"] == "preempted"  # settled once
    nv.reserve(tmp_path, {"run_id": "s", "state": "reserved", "charged_bound_usd": 1.0, "rate_usd_per_hour": 1.254})
    assert nv.settle_entry(tmp_path, "s", "complete", 10 ** 6)["charged_bound_usd"] == 1.0
    nv.reserve(tmp_path, {"run_id": "t", "state": "reserved", "charged_bound_usd": 1.0, "rate_usd_per_hour": 1.254})
    assert nv.settle_entry(tmp_path, "t", "launch_failed_no_execution", 0, bound=0.0)["charged_bound_usd"] == 0.0


def test_real_ledger_survives_a_round_trip(tmp_path):
    source = SCRIPTS.parent / "docs/spend.json"
    if not source.is_file():  # a generated, untracked view since 24 Oct (absent in a fresh checkout / CI)
        pytest.skip("no local docs/spend.json")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/spend.json").write_bytes(source.read_bytes())
    before = json.loads(source.read_text()).get("nebius")
    nv.set_cap(tmp_path, 10, "test")
    with nv.ledger(tmp_path) as data:
        if before is None:
            del data["nebius"]
        else:
            data["nebius"] = before
    assert (tmp_path / "docs/spend.json").read_bytes() == source.read_bytes()


# -- rendering ---------------------------------------------------------------------------------------------------
def spec_for(run_id="b2-n1"):
    return {"run_id": run_id, "bucket": "arc-bucket", "launch_prefix": f"launch/{run_id}", "runs_prefix": "runs",
            "deadline_epoch": 1791400000.7, "runner_sha256": "ab" * 32}


def test_bootstrap_renders_and_parses(tmp_path):
    config = nv.check_config(CONFIG)
    script = nv.render_bootstrap(spec_for(), config)
    assert "@" not in "".join(line for line in script.splitlines() if line.startswith(("RUN_ID", "BUCKET", "DEADLINE")))
    assert "DEADLINE_EPOCH=1791400000" in script and "SECRET_ID=mbsec-e00x" in script
    path = tmp_path / "bootstrap.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True).returncode == 0
    # The deadline guard is armed before anything that can fail, and the run unit deletes the VM when it ends.
    assert script.index("arc-deadline") < script.index("apt-get") < script.index("mysterybox payload get")
    assert "ExecStopPost=" in script and "arc-self-delete" in script.split("ExecStopPost=")[1].splitlines()[0]
    assert "compute instance delete --id" in script and "shutdown -h now" in script
    assert "KillMode=mixed" in script
    injected = nv.render_bootstrap({**spec_for(), "run_id": "x; rm -rf /"}, config)
    assert "RUN_ID='x; rm -rf /'" in injected


def test_user_data_carries_bootstrap_and_spec_without_secrets():
    config = nv.check_config({**CONFIG, "ssh_user": "richard", "ssh_public_key": "ssh-ed25519 AAAA test"})
    spec = spec_for()
    text = nv.render_user_data(spec, config)
    assert text.startswith("#cloud-config\n") and "ssh-ed25519 AAAA test" in text
    assert text.rstrip().endswith("- [bash, /opt/arc/bootstrap.sh]")
    blobs = [base64.b64decode(line.split("content: ")[1]).decode() for line in text.splitlines()
             if line.strip().startswith("content: ")]
    assert blobs[0] == nv.render_bootstrap(spec, config) and json.loads(blobs[1]) == spec
    assert "AWS_SECRET_ACCESS_KEY=" not in blobs[0].replace('print(f"{name}={found[name]}")', "")
    assert "users:" not in nv.render_user_data(spec, nv.check_config(CONFIG))


def test_create_command_flags():
    config = nv.check_config(CONFIG)
    argv = nv.create_command(config, "arc-b2-n1", {"arc-run": "b2-n1", "arc-deadline": "1791400000"}, "UD")

    def value(flag, args=None):
        args = args or argv
        return args[args.index(flag) + 1]

    assert argv[:3] == ["compute", "instance", "create"] and argv[-2:] == ["--cloud-init-user-data", "UD"]
    assert value("--resources-platform") == "gpu-rtx6000-a" and value("--resources-preset") == "1gpu-24vcpu-218gb"
    assert value("--boot-disk-managed-disk-type") == "network_ssd_non_replicated"
    assert value("--boot-disk-managed-disk-size-gibibytes") == "744"
    assert value("--boot-disk-managed-disk-source-image-family-image-family") == "ubuntu24.04-cuda13.0"
    assert value("--recovery-policy") == "fail" and value("--preemptible-on-preemption") == "stop"
    assert value("--spot-pricing-policy-id") == "pricingpolicy-e00x" and "--follows-spot-price" not in argv
    assert value("--labels") == "arc-run=b2-n1,arc-deadline=1791400000"
    assert json.loads(value("--network-interfaces")) == [{"name": "eth0", "ip_address": {}, "public_ip_address": {},
                                                          "subnet_id": "vpcsubnet-e00x"}]
    follow = nv.check_config({**CONFIG, "pricing_policy_id": None, "spot_max_price_usd_per_gpu_hour": None,
                              "follow_spot_price": True})
    assert "--follows-spot-price" in nv.create_command(follow, "n", {}, "UD")
    regular = nv.create_command(config, "n", {}, "UD", preemptible=False)
    assert "--on-demand" in regular and "--preemptible-on-preemption" not in regular
    assert "--spot-pricing-policy-id" not in regular


def test_bundle_archive_skips_what_modal_skips(tmp_path):
    bundle = make_bundle(tmp_path)
    digest = nv.pack_bundle(bundle, tmp_path / "b.tar.gz")
    assert digest == hashlib.sha256((tmp_path / "b.tar.gz").read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "b.tar.gz") as tar:
        names = {n.removeprefix("./") for n in tar.getnames()}
    assert {"MANIFEST.json", "bootstrap_arena.py", "configs/comparison.json"} <= names
    assert not any("dataset-upload" in n or n.endswith(".ipynb") for n in names)
    (bundle / "bootstrap_arena.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        nv.verify_bundle(bundle)


# -- VM side: paths, environment, manifests ----------------------------------------------------------------------
def test_job_command_and_paths_match_the_modal_container():
    spec = {"script": "bootstrap_arena.py", "timeout": 86400, "arguments": ["--trial", "duck-transfer"],
            "block_network": False}
    output = Path("/runs/validation/b2-n1")
    assert runner.job_command(spec, output, python="/usr/bin/python3") == [
        "/usr/bin/python3", "/opt/arc-validation/bootstrap_arena.py", "--bundle", "/opt/arc-validation",
        "--working", "/tmp/arena", "--output", "/runs/validation/b2-n1/arena", "--games-dir", "/assets/games",
        "--max-seconds", "86340", "--trial", "duck-transfer"]
    isolated = runner.job_command({**spec, "block_network": True}, output, python="/usr/bin/python3")
    assert isolated[:3] == ["unshare", "--net", "--"] and isolated[-14:] == runner.job_command(
        spec, output, python="/usr/bin/python3")
    assert "ip link set lo up" in isolated[5]


def test_job_environment_matches_modal_and_drops_credentials():
    base = {"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "s", "AWS_ACCESS_KEY_ID": "k", "NEBIUS_IAM_TOKEN": "t",
            "RCLONE_CONFIG_ARC_TYPE": "s3", "LIBRARY_PATH": "/x"}
    env = runner.job_environment(base, {"kind": "arena", "test": "b200-run"}, Path("/runs/validation/r"))
    assert not any(k.startswith(("AWS_", "RCLONE_", "NEBIUS_")) for k in env)
    assert env["PATH"] == "/usr/bin" and env["LIBRARY_PATH"].startswith("/usr/local/nvidia/lib64:")
    expected = {"PYTHONPATH": "/opt/arc-validation/src", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1", "MPLBACKEND": "Agg", "ONLY_RESET_LEVELS": "true",
                "PYTHONDONTWRITEBYTECODE": "1", "ARC_FORCE_CHROOT": "1", "ARENA_STOP_REQUEST": "/tmp/arena-stop-request", "ARENA_CONTROL": "/tmp/arena-control.json", "ARENA_KNOBS": "/tmp/arena-knobs.json",
                "ARC_VALIDATION_OUTPUT": "/runs/validation/r", "ARC_VALIDATION_KIND": "arena",
                "ARC_VALIDATION_TEST": "b200-run"}
    assert {k: env[k] for k in expected} == expected
    # The same names the Modal launcher sets in its container.
    modal_source = (SCRIPTS / "modal_validation.py").read_text()
    for name in ("HF_HUB_OFFLINE", "ONLY_RESET_LEVELS", "ARC_FORCE_CHROOT", "ARENA_STOP_REQUEST", "/assets/games",
                 "/tmp/arena", "/opt/arc-validation", "/assets/flash-input"):
        assert name in modal_source


def test_stop_watcher_mirrors_both_live_files(tmp_path, monkeypatch):
    """live knobs: arena/live/control.json -> $ARENA_CONTROL and knobs.json -> $ARENA_KNOBS each poll; the
    stop request still ends the watch."""
    import threading
    import types

    polls = []
    for name, value in (("STOP_LOCAL", "stop"), ("CONTROL_LOCAL", "control"), ("KNOBS_LOCAL", "knobs")):
        monkeypatch.setattr(runner, name, str(tmp_path / value))

    def rclone(*args, timeout=None, check=True):
        if args[0] == "lsf":
            polls.append(1)
            return types.SimpleNamespace(stdout="control.json\nknobs.json\n" + ("stop-request\n" if len(polls) >= 2 else ""))
        Path(args[2]).write_text(f"{args[1].rsplit('/', 1)[-1]} {len(polls)}")
        return types.SimpleNamespace(stdout="")

    fake = types.SimpleNamespace(remote_runs="bucket:runs/validation/r", spec={"stop_poll_seconds": 0.01},
                                 stop=threading.Event(), output=tmp_path / "out", rclone=rclone)
    runner.Runner.stop_watcher(fake)
    assert (tmp_path / "control").read_text() == "control.json 2" and (tmp_path / "knobs").read_text() == "knobs.json 2"
    assert (tmp_path / "stop").exists() and len(polls) == 2


def test_live_knobs_mirror_replaces_atomically_and_follows_deletes(tmp_path):
    """arc3sub 30 Oct (checklist item 5): arena/live/knobs.json -> $ARENA_KNOBS; each new version replaces the file by
    rename, an unchanged one is not rewritten, a failed copy keeps the last good file, a deleted remote removes it."""
    import types

    remote = {"body": b'{"active_limit": 40}'}
    local = tmp_path / "arena-knobs.json"

    def rclone(*args, timeout=None, check=True):
        assert args[0] == "copyto" and args[2] == f"{local}.tmp"
        if remote["body"] is None:
            raise subprocess.TimeoutExpired("rclone", 60)
        Path(args[2]).write_bytes(remote["body"])
        return types.SimpleNamespace(stdout="")

    runner.mirror_live_file(rclone, "b:k", str(local), True)
    assert local.read_bytes() == b'{"active_limit": 40}' and not Path(f"{local}.tmp").exists()
    inode = local.stat().st_ino
    runner.mirror_live_file(rclone, "b:k", str(local), True)  # unchanged: same file
    assert local.stat().st_ino == inode
    remote["body"] = b'{"active_limit": 64}'
    runner.mirror_live_file(rclone, "b:k", str(local), True)
    assert local.read_bytes() == b'{"active_limit": 64}'
    remote["body"] = None  # a failed copy: the last good knobs stay
    runner.mirror_live_file(rclone, "b:k", str(local), True)
    assert local.read_bytes() == b'{"active_limit": 64}' and not Path(f"{local}.tmp").exists()
    runner.mirror_live_file(rclone, "b:k", str(local), False)
    assert not local.exists()


@pytest.fixture(autouse=True)
def keep_signal_handlers():
    """Runner.execute installs on_signal for its process; on a shared xdist worker that broke test_runops_restart's SIGINT test."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


def test_calibration_host_bench_and_a_missing_gpu_only_log(tmp_path):
    """identity and fixed benches before the assets. No driver here: the GPU stage is an error line."""
    got = runner.calib_bench_host(per_thread=1 << 20, repeats=2, loops=1000)
    assert got["host_mem_copy_gbs"] > 0 and got["host_mem_copy_1t_gbs"] > 0 and got["host_py_mops"] > 0
    assert got["host_py_check"] == runner.calib_bench_host(per_thread=1 << 20, repeats=1, loops=1000)["host_py_check"]
    path = tmp_path / "calibration.json"
    data = runner.calibrate(str(path))
    assert json.loads(path.read_text())["schema"] == runner.CALIBRATE_SCHEMA and "host" in data
    assert any("calib_bench_gpu" in e for e in data["errors"]) and data["bench"]["host_py_mops"] > 0


def test_calibration_gpu_bench_calls_the_driver_in_order():
    import ctypes

    calls = []

    class Fake:
        def __getattr__(self, name):
            def call(*args):
                calls.append(name)
                if name == "cuMemAlloc_v2":
                    args[0]._obj.value = 1 << 40
                if name == "cuMemcpyDtoH_v2":
                    args[0]._obj.value = 2.0
                if name == "cuLaunchKernel":
                    assert args[1].value * args[4].value >= 1024 and args[4].value == 256 and args[8] is None
                    assert ctypes.cast(args[9][0], ctypes.POINTER(ctypes.c_ulonglong)).contents.value == 1 << 40
                return 0
            return call

    got = runner.calib_bench_gpu(Fake(), core=(1024, 4), mem_elements=1 << 12, repeats=3)
    assert calls[:3] == ["cuInit", "cuDeviceGet", "cuCtxCreate_v2"] and calls.count("cuLaunchKernel") == 6
    assert calls.index("cuModuleLoadData") < calls.index("cuLaunchKernel")
    assert all(got[k] > 0 for k in ("gpu_core_gflops", "gpu_mem_gbs", "gpu_fill_gbs", "gpu_h2d_gbs"))
    assert b".entry calib" in runner.CALIB_PTX and runner.CALIB_PTX.endswith(b"\x00")

    class Bad(Fake):
        def __getattr__(self, name):
            return (lambda *a: 200) if name == "cuModuleLoadData" else super().__getattr__(name)

    with pytest.raises(RuntimeError, match="cuModuleLoadData: CUDA error 200"):
        runner.calib_bench_gpu(Bad(), core=(1024, 4), mem_elements=1 << 12, repeats=3)


def test_calibration_never_blocks_the_run(tmp_path, monkeypatch):
    import types

    saved = []
    fake = types.SimpleNamespace(output=tmp_path, run_id="r", spec={"attempt": 2, "calibrate_seconds": 0.3}, instance=lambda: "i",
                                 put_json=lambda name, data: saved.append((name, data)))
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    assert runner.Runner.calibrate_vm(fake) is None and not saved  # no GPU: nothing to do
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    real = subprocess.Popen
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **k: real([sys.executable, "-c", "import time; time.sleep(60)"], **k))
    began = time.time()
    out = runner.Runner.calibrate_vm(fake)  # a hung bench is killed at its limit
    assert time.time() - began < 10 and out["errors"] == 1 and "limit" in saved[0][1]["errors"][0]

    def broken(*a, **k):
        raise OSError("no fork")
    monkeypatch.setattr(runner.subprocess, "Popen", broken)
    assert runner.Runner.calibrate_vm(fake)["errors"] == 1 and "no fork" in saved[1][1]["errors"][0]  # never raises
    fake.put_json = broken
    assert runner.Runner.calibrate_vm(fake)["errors"] == 1  # nor when the upload fails


def test_manifest_verification(tmp_path):
    root = tmp_path / "assets"
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a" * 100)
    (root / "sub/b.bin").write_bytes(b"b" * 50)
    manifest = {"prefix": "p", "files": {
        "a.bin": {"size": 100, "sha256": hashlib.sha256(b"a" * 100).hexdigest()},
        "sub/b.bin": {"size": 50, "sha256": hashlib.sha256(b"b" * 50).hexdigest()}}}
    assert runner.verify_manifest(root, manifest)["bytes"] == 150
    (root / "sub/b.bin").write_bytes(b"c" * 50)
    assert runner.verify_manifest(root, manifest, "size")["files"] == 2
    with pytest.raises(ValueError, match="sha256 mismatch: p/sub/b.bin"):
        runner.verify_manifest(root, manifest)
    (root / "a.bin").write_bytes(b"a" * 99)
    with pytest.raises(ValueError, match="size mismatch: p/a.bin"):
        runner.verify_manifest(root, manifest, "size")
    assert runner.asset_slug("flash-input/x/y") == nv.asset_slug("flash-input/x/y")


def test_stop_rule():
    rule = {"score_target": 95.0, "usd_limit": 20}
    assert runner.stop_rule_reason(rule, 96.0, 1) and runner.stop_rule_reason(rule, None, 20.5)
    assert runner.stop_rule_reason(rule, 50.0, 19.9) is None
    assert runner.stop_rule_reason({"score_target": 95.0, "usd_limit": None}, None, 10 ** 6) is None


# -- states ------------------------------------------------------------------------------------------------------
def test_classify_and_reap_reasons():
    now, deadline = 1000.0, 2000.0
    status, final = {"phase": "bootstrap"}, {"state": "complete"}
    assert nv.classify("RUNNING", None, None, None, now, deadline) == "starting"
    assert nv.classify("RUNNING", status, None, None, now, deadline) == "running"
    assert nv.classify("RUNNING", status, None, None, 2400, deadline) == "overdue"
    assert nv.classify(None, status, final, None, now, deadline) == "complete"
    assert nv.classify(None, status, {"state": "interrupted"}, None, now, deadline) == "preempted"
    assert nv.classify(None, status, {"state": "interrupted"}, {"reason": "operator"}, now, deadline) == "reaped"
    assert nv.classify("STOPPED", status, None, None, now, deadline) == "preempted"
    assert nv.classify(None, status, None, None, now, deadline) == "lost"
    assert nv.classify(None, None, {"state": "setup_failed"}, None, now, deadline) == "setup_failed"
    labels = {"arc-run": "r", "arc-deadline": "2000"}
    assert nv.reap_reason(labels, "RUNNING", None, now) is None
    assert "past its deadline" in nv.reap_reason(labels, "RUNNING", None, 2100)
    assert "stopped" in nv.reap_reason(labels, "STOPPED", None, now)
    assert "finished" in nv.reap_reason(labels, "RUNNING", final, now)
    assert nv.reap_reason({"arc-run": "r"}, "RUNNING", None, now) == "no deadline label"


# -- launch, dry run, preemption bookkeeping -----------------------------------------------------------------------
def test_dry_run_prints_every_cloud_call_and_touches_nothing(tmp_path, monkeypatch):
    path = make_ledger(tmp_path, cap=2000)
    before = path.read_text()
    bundle = make_bundle(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("dry run made a call"))
    lines = []
    cloud = nv.Cloud(nv.check_config(CONFIG), dry_run=True, echo=lines.append)
    launch_spot(tmp_path, cloud.config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=bundle)
    text = "\n".join(lines)  # (a dry run needs no approval: nothing is launched)
    assert all(line.startswith("DRY-RUN ") for line in lines)
    assert "rclone lsf nebius-arc:arc-bucket/assets-manifests/" in text
    assert "nebius-arc:arc-bucket/launch/b2-n1/bundle.tar.gz" in text
    assert "nebius-arc:arc-bucket/launch/b2-n1/nebius_runner.py" in text
    assert "compute instance create --parent-id project-e00x --name arc-b2-n1" in text
    assert "--spot-pricing-policy-id pricingpolicy-e00x" in text and "--format json" in text
    assert "ledger: reserve $" in text and "receipt:" in text and "status.json" in text
    assert path.read_text() == before and not (tmp_path / nv.RECORDS).exists()
    kept = Path(text.split("$(cat ")[1].split(")")[0])
    assert kept.read_text().startswith("#cloud-config")


def test_launch_reserves_then_creates_and_records(tmp_path):
    make_ledger(tmp_path, cap=2000)
    bundle = make_bundle(tmp_path)
    config = nv.check_config(CONFIG)
    assets = nv.required_assets(STUDY)
    cloud = FakeCloud(config, manifests(assets[:-1]))
    with pytest.raises(ValueError, match="more than 4 h of play needs the user's approval"):
        launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=bundle)
    with pytest.raises(RuntimeError, match="not mirrored"):
        launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=bundle,
                  approval=APPROVED)
    assert "nebius" in json.loads((tmp_path / "docs/spend.json").read_text())
    assert not json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"]

    objects = {**manifests(assets), "runs/validation/b2-n1/nebius/status.json": {"phase": "boot"}}
    cloud = FakeCloud(config, objects)
    record = launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=bundle,
                       now=1_000_000.0, approval=APPROVED)
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    seconds = 86400 + 2400 + 300
    # The whole run is reserved at launch: this bundle cannot resume, so each of the 3 relaunches may redo it all.
    whole = seconds + 3 * (900 + 86400)
    assert entry["platform"] == "nebius" and entry["attempt_seconds"] == seconds and entry["preemptible"] is True
    assert entry["max_seconds"] == whole and entry["max_attempts"] == 4 and entry["long_run_approval"] == APPROVED
    assert entry["charged_bound_usd"] == pytest.approx(nv.bound_usd(whole, 1.2 + 744 * 0.053 / 730))
    assert record["policy"]["checkpointing"] is False and cloud.objects["launch/b2-n1/record.json"]["attempts"]
    assert entry["rate_usd_per_hour"] == pytest.approx(1.2 + 744 * 0.053 / 730)
    saved = nv.read_record(tmp_path, "b2-n1")
    assert saved["state"] == "submitted" and saved["instance_id"] == "computeinstance-e00new"
    assert saved["deadline_epoch"] == 1_000_000.0 + seconds and saved["platform"] == "nebius"
    assert record["outputs"] == "s3://arc-bucket/runs/validation/b2-n1"
    order = [" ".join(c[:4]) for c in cloud.calls]
    assert order.index(next(c for c in order if "copyto" in c)) < order.index(
        next(c for c in order if "compute instance create" in c))


def test_no_heartbeat_deletes_the_vm(tmp_path, monkeypatch):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "heartbeat_seconds": 0})
    cloud = FakeCloud(config, manifests(nv.required_assets(STUDY)))
    with pytest.raises(RuntimeError, match="No heartbeat"):
        launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=make_bundle(tmp_path),
                  approval=APPROVED)
    assert cloud.deleted == ["computeinstance-e00new"]
    assert nv.read_record(tmp_path, "b2-n1")["state"] == "failed_no_heartbeat"
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "failed_no_heartbeat" and entry["charged_bound_usd"] < 1


def spec_of(cloud, index=-1):
    """The launch.json of the index-th created VM (its cloud-init carries it)."""
    created = [c for c in cloud.calls if "create" in c and "compute" in c]
    text = created[index][created[index].index("--cloud-init-user-data") + 1]
    blobs = [line.split("content: ")[1] for line in text.splitlines() if "content: " in line]
    return json.loads(base64.b64decode(blobs[1]))


def preempt(cloud, run_id, ended, checkpoint=None):
    base = f"runs/validation/{run_id}/nebius"
    cloud.objects[f"{base}/final.json"] = {"state": "interrupted", "ended_epoch": ended,
                                           "interrupted": {"signal": "SIGTERM", "epoch": ended - 10}}
    if checkpoint:
        cloud.objects[f"{base}/checkpoint.json"] = {"trials": {"duck-transfer": checkpoint}, "done": []}
    cloud.state = "STOPPED"


def test_a_preempted_run_resumes_from_its_checkpoint_in_the_same_folder(tmp_path):
    make_ledger(tmp_path, cap=200)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0, approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["max_seconds"] == 89100 + 3 * (900 + 600 + 120)  # a resumable bundle loses one interval per relaunch
    first = spec_of(cloud)
    assert first["attempt"] == 1 and first["checkpoint"] == {"seconds": 600} and first["deadline_epoch"] == 90100.0
    info, _ = nv.advance(tmp_path, config, cloud, "b2-n1", now=5000.0)
    assert info["state"] == "running" and len([c for c in cloud.calls if "create" in c]) == 1

    preempt(cloud, "b2-n1", 8200.0, {"seq": 12, "elapsed_seconds": 6500.0})
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=8300.0)
    assert info["state"] == "relaunched" and cloud.deleted == ["computeinstance-e00new"]
    assert [(a["run_id"], a["kind"], a.get("state")) for a in record["attempts"]] == [
        ("b2-n1", "fresh", "preempted"), ("b2-n1", "resumed", None)]
    assert record["attempts"][0]["checkpoint"] == {"duck-transfer": {"seq": 12, "elapsed_seconds": 6500.0}}
    assert record["attempts"][0]["observed_seconds"] == 7200 and record["state"] == "submitted"
    second = spec_of(cloud)
    assert second["run_id"] == "b2-n1" and second["attempt"] == 2 and second["launch_prefix"] == "launch/b2-n1"
    left = 89100 + 3 * 1620 - 7200  # what the reservation still covers bounds the second machine
    assert second["deadline_epoch"] == 8300.0 + left and second["timeout"] == left - 2400 - 300
    assert record["instance_id"] == "computeinstance-e00new2" and record["instance_name"] == "arc-b2-n1-2"
    # The first machine's markers moved aside; the ledger still holds the whole-run reservation.
    assert "runs/validation/b2-n1/nebius/final.json" not in cloud.objects
    assert cloud.objects["runs/validation/b2-n1/nebius/attempts/1/final.json"]["state"] == "interrupted"
    assert json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]["state"] == "reserved"
    assert cloud.objects["launch/b2-n1/record.json"]["attempts"][1]["kind"] == "resumed"

    cloud.state = None  # the second VM finished and deleted itself
    cloud.objects["runs/validation/b2-n1/nebius/final.json"] = {"state": "complete", "ended_epoch": 8300.0 + 5000}
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=14000.0)
    assert info["state"] == "complete" and record["state"] == "complete" and record["observed_seconds"] == 12200
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "complete" and entry["observed_seconds"] == 12200
    assert entry["charged_bound_usd"] == pytest.approx((12200 + 120) * entry["rate_usd_per_hour"] / 3600 * 1.15)


def test_without_a_checkpoint_the_relaunch_starts_from_zero_under_a_new_id_and_only_after_preemptions(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "max_attempts": 3})
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    bundle = make_bundle(tmp_path, checkpointing=True)
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=bundle, now=1000.0,
              approval=APPROVED)
    preempt(cloud, "b2-n1", 1500.0)  # cut before the first checkpoint reached the bucket
    info, record = nv.inspect(tmp_path, cloud, "b2-n1", now=1600.0)
    assert info["state"] == "relaunch_pending" and record["pending"]["kind"] == "fresh"

    class NoCapacity(FakeCloud):
        def run(self, argv, **kwargs):
            if "create" in argv:
                raise RuntimeError("compute instance create exited 1: not enough resources")
            return super().run(argv, **kwargs)

    dry = NoCapacity(config, cloud.objects)
    info, record = nv.advance(tmp_path, config, dry, "b2-n1", now=1700.0)  # no spot capacity: stays pending
    assert record["state"] == "relaunch_pending" and record["pending"]["failures"] == 1
    cloud.state = "RUNNING"
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=1800.0)
    assert info["state"] == "relaunched" and record["attempts"][1]["run_id"] == "b2-n1-a2"
    second = spec_of(cloud)
    assert second["run_id"] == "b2-n1-a2" and second["attempt"] == 1  # a new folder, nothing to resume
    assert record["outputs"].endswith("/runs/validation/b2-n1-a2")
    assert "runs/validation/b2-n1/nebius/final.json" in cloud.objects  # the partial run stays as it ended

    # A failure that is not a preemption ends the run: no third attempt.
    cloud.state = None
    cloud.objects["runs/validation/b2-n1-a2/nebius/final.json"] = {"state": "failed", "ended_epoch": 2800.0}
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=2900.0)
    assert info["state"] == "failed" and record["state"] == "failed" and len(record["attempts"]) == 2
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "failed" and entry["observed_seconds"] == 500 + 1000

    # A checkpoint the trial could not continue is the one failure that restarts (from zero); the limit holds.
    record = {"policy": {"auto_relaunch": True, "max_attempts": 3, "checkpointing": True}, "attempts": [{}, {}]}
    assert nv.decide(record, "failed", {"resume_failed": True}, None) == "fresh"
    assert nv.decide(record, "preempted", None, {"trials": {}, "done": ["sanity"]}) == "resume"
    assert nv.decide(record, "timeout", None, {"trials": {"t": {}}}) is None
    assert nv.decide({**record, "attempts": [{}, {}, {}]}, "preempted", None, {"trials": {"t": {}}}) is None
    assert nv.decide({"attempts": [{}]}, "preempted", None, None) is None  # a receipt from before the policy


def test_no_spot_capacity_for_the_whole_window_ends_the_run_as_relaunch_failed(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "capacity_retry_seconds": 1800})
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0, approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    preempt(cloud, "b2-n1", 5000.0, {"seq": 4, "elapsed_seconds": 2400.0})

    class NoCapacity(FakeCloud):
        def run(self, argv, **kwargs):
            if "create" in argv:
                raise RuntimeError("compute instance create exited 1: not enough resources")
            return super().run(argv, **kwargs)

    dry = NoCapacity(config, cloud.objects, instance_state="STOPPED")
    info, record = nv.advance(tmp_path, config, dry, "b2-n1", now=5100.0)
    assert record["state"] == "relaunch_pending" and "not enough resources" in record["pending"]["last_error"]
    info, record = nv.advance(tmp_path, config, dry, "b2-n1", now=5100.0 + 1700)  # still inside the window
    assert record["state"] == "relaunch_pending" and record["pending"]["failures"] == 2
    info, record = nv.advance(tmp_path, config, dry, "b2-n1", now=5100.0 + 1900)
    assert info["state"] == "relaunch_failed" and record["state"] == "relaunch_failed"
    assert len(record["attempts"]) == 1 and not [c for c in dry.calls if "--on-demand" in c]  # no fallback
    assert "ask the user" in record["next"]  # 82,200 s of play: over 4 h
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "relaunch_failed" and entry["observed_seconds"] == 4000
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=9000.0)  # over: nothing starts later
    assert record["state"] == "relaunch_failed" and len(record["attempts"]) == 1


def test_the_long_run_rule_counts_play_time(tmp_path):
    duck = {"name": "duck-transfer", "kind": "duck-transfer"}
    trial = ["--trial", "duck-transfer"]
    study = {"wall_seconds": 7200, "smoke": {"wall_seconds": 120}, "trials": [duck]}
    assert nv.play_seconds(study, trial, 9900) == 7200 and nv.play_seconds(study, ["--smoke"], 1680) == 120
    assert nv.play_seconds({**study, "competition": {"preview_seconds": 3600}}, trial, 7800) == 3600
    assert nv.play_seconds({**study, "early_stop": {"always": True, "at_seconds": 1800}}, trial, 9900) == 1800
    assert nv.play_seconds({**study, "trials": [{**duck, "wall_seconds": 900}, duck]}, [], 9900) == 8100
    # A trial that plays whatever the timeout leaves, or states no wall, counts as the timeout.
    assert nv.play_seconds({"trials": [{"name": "level-bench", "kind": "level-bench"}]}, [], 13500) == 13500
    assert nv.play_seconds({"trials": [duck]}, trial, 5000) == 5000

    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "capacity_retry_seconds": 1800})
    sibling = tmp_path / "modal_validation.py"
    sibling.write_text("def validation_timeout(kind, test):\n    return int(test.split('-')[1]) + 2700\n"
                       "def arena_arguments(test):\n    return ['--trial', 'duck-transfer']\n")
    os.environ["ARC_MODAL_VALIDATION"] = str(sibling)
    try:
        def launch(wall, **kwargs):
            run = f"r{wall}-{len(kwargs)}"
            cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                                       f"runs/validation/{run}/nebius/status.json": {"phase": "boot"}})
            bundle = make_bundle(tmp_path, {**STUDY, "wall_seconds": wall, "b200": None}, checkpointing=True)
            return launch_spot(tmp_path, config, cloud, run_id=run, kind="arena", test=f"play-{wall}", bundle=bundle,
                             **kwargs)

        record = launch(7200)  # 2 h of play in a 9,900 s timeout: no flag
        assert record["play_seconds"] == 7200 and "long_run_reason" not in record
        assert record["policy"]["capacity_retry_seconds"] == 1800
        with pytest.raises(ValueError, match="more than 2 h of play needs --reason"):
            launch(9000)
        record = launch(9000, reason="the late-score mechanism needs hour 3")
        assert record["long_run_reason"] == "the late-score mechanism needs hour 3"
        entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][-1]
        assert entry["long_run_reason"] == record["long_run_reason"] and entry["play_seconds"] == 9000
        with pytest.raises(ValueError, match="more than 4 h of play needs the user's approval"):
            launch(15000, reason="not enough above 4 h")
        assert launch(15000, approval=APPROVED)["long_run_approval"] == APPROVED
    finally:
        del os.environ["ARC_MODAL_VALIDATION"]


def test_relaunches_stay_inside_the_reservation(tmp_path):
    config = nv.check_config(CONFIG)
    bundle = make_bundle(tmp_path, checkpointing=True)
    (bundle / "src/arena/checkpoint.py").write_text('UNSUPPORTED = ("explorer", "new_feature")\n')
    trial = ["--trial", "duck-transfer"]
    assert nv.resumable(bundle, STUDY, trial) == (True, True) and nv.resumable(bundle, STUDY, []) == (True, True)
    # What cannot be cut restarts from zero, and is reserved as such: the bundle's own list of features, other kinds.
    assert nv.resumable(bundle, {**STUDY, "new_feature": {"on": 1}}, trial) == (True, False)
    bench = {**STUDY, "trials": [*STUDY["trials"], {"name": "level-bench", "kind": "level-bench"}]}
    assert nv.resumable(bundle, bench, trial) == (True, True) and nv.resumable(bundle, bench, []) == (True, False)
    assert nv.resumable(bundle, bench, ["--trial", "level-bench"]) == (True, False)
    assert nv.resumable(make_bundle(tmp_path), STUDY, trial) == (False, False)
    assert nv.reserved_seconds(10000, 7200, config, checkpointing=True, max_attempts=4) == (10000 + 3 * 1620, 1620)
    assert nv.reserved_seconds(10000, 7200, config, checkpointing=False, max_attempts=2) == (10000 + 8100, 8100)
    assert nv.reserved_seconds(10000, 7200, config, checkpointing=True, max_attempts=1) == (10000, 1620)
    cloud = FakeCloud(config)
    record = {"run_id": "r", "test": "t", "preemptible": True, "reserved_seconds": 12000, "attempt_seconds": 10000,
              "spec": {"timeout": 7300, "runs_prefix": "runs", "bucket": "arc-bucket", "launch_prefix": "launch/r",
                       "runner_sha256": "0" * 64},
              "attempts": [{"n": 1, "run_id": "r", "segment": 1, "observed_seconds": 6000}]}
    attempt = nv.start_attempt(config, cloud, record, "resume", 50_000.0)
    assert attempt["deadline_epoch"] == 50_000.0 + 6000 and spec_of(cloud)["timeout"] == 6000 - 2400 - 300
    record["attempts"][-1]["observed_seconds"] = 5000
    with pytest.raises(RuntimeError, match="reservation used"):
        nv.start_attempt(config, cloud, record, "resume", 60_000.0)


def test_the_controller_relaunches_from_bucket_records_and_the_laptop_stands_back(tmp_path):
    make_ledger(tmp_path, cap=200)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    before = (tmp_path / "docs/spend.json").read_text()
    preempt(cloud, "b2-n1", time.time() + 100, {"seq": 3, "elapsed_seconds": 60.0})
    remote = nv.check_config({**CONFIG, "rclone_remote": "arc", "nebius_binary": "/usr/local/bin/arc-nebius"})
    controller = FakeCloud(remote, cloud.objects, instance_state="STOPPED")
    controller.created = 1
    assert nv.controller_loop(remote, controller, rounds=1) == "rounds"
    beat = cloud.objects["launch/controller/heartbeat.json"]
    assert beat["active"] == [{"run_id": "b2-n1", "state": "relaunched"}] and nv.controller_alive(cloud)
    assert [c for c in controller.calls if "create" in c][0][0] == "/usr/local/bin/arc-nebius"
    assert spec_of(controller)["attempt"] == 2 and (tmp_path / "docs/spend.json").read_text() == before
    # Here, the receipt follows the bucket's; a pending relaunch is left to the live controller.
    cloud.state = "RUNNING"
    cloud.objects["runs/validation/b2-n1/nebius/status.json"] = {"phase": "assets"}
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "running" and [a["kind"] for a in record["attempts"]] == ["fresh", "resumed"]
    preempt(cloud, "b2-n1", time.time() + 200, {"seq": 5, "elapsed_seconds": 90.0})
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "relaunch_pending" and len(record["attempts"]) == 2
    # A controller that keeps failing (its account may not create VMs) does not hold the relaunch back for long.
    record["pending"]["failures"] = 3
    nv.save_record(tmp_path, cloud, record)
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "relaunched" and len(record["attempts"]) == 3
    preempt(cloud, "b2-n1", time.time() + 300, {"seq": 7, "elapsed_seconds": 120.0})
    cloud.objects["launch/controller/heartbeat.json"]["epoch"] -= 3600  # the controller is gone: this side acts
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "relaunched" and len(record["attempts"]) == 4
    # An idle controller leaves (its unit's end deletes the VM); finished and policy-less runs are not its business.
    cloud.objects["launch/b2-n1/record.json"]["state"] = "complete"
    assert nv.controller_loop(remote, controller, rounds=3, interval=0, idle_seconds=0) == "idle"


def test_every_vm_installs_this_sides_cli_version(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
              bundle=make_bundle(tmp_path))
    created = next(c for c in cloud.calls if "create" in c)
    text = created[created.index("--cloud-init-user-data") + 1]
    script = base64.b64decode(next(line.split("content: ")[1] for line in text.splitlines()
                                   if "content: " in line)).decode()
    assert "\nCLI_VERSION=0.12.286\n" in script and "cli/release/$CLI_VERSION/linux/$(uname -m)/nebius" in script
    assert 'BIN=/opt/arc/bin/nebius; [ -x "$BIN" ] || BIN=nebius' in script  # the wrapper prefers the installed one
    # Without a readable local version nothing is pinned: the image's CLI or the installer, as before.
    assert "\nCLI_VERSION=''\n" in nv.render_bootstrap(spec_for(), nv.check_config(CONFIG))
    old = FakeCloud(nv.check_config(CONFIG))
    old.lacking = ("--spot-pricing-policy-id",)
    assert old.missing_create_flags() == ["--spot-pricing-policy-id"] and cloud.missing_create_flags() == []
    old.lacking = ("--parent-id",)
    assert old.missing_create_flags() is None  # no usable help text at all


def test_an_unfit_controller_says_so_and_the_laptop_relaunches(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    preempt(cloud, "b2-n1", time.time() + 100, {"seq": 1, "elapsed_seconds": 92.66})
    remote = nv.check_config({**CONFIG, "rclone_remote": "arc", "nebius_binary": "/usr/local/bin/arc-nebius",
                              "cli_version": "0.12.286"})
    controller = FakeCloud(remote, cloud.objects, instance_state="STOPPED")
    controller.version, controller.lacking, controller.created = "0.12.101", ("--spot-pricing-policy-id",), 1
    assert nv.controller_loop(remote, controller, rounds=2, interval=0) == "rounds"
    beat = cloud.objects["launch/controller/heartbeat.json"]
    assert beat["state"] == "controller_unfit" and beat["unfit"] == {
        "reason": "cli", "missing_flags": ["--spot-pricing-policy-id"], "cli_version": "0.12.101",
        "wanted": "0.12.286"}
    # It never tried a create, but it closed the preempted attempt (VM deleted) and left the relaunch pending.
    assert not [c for c in controller.calls if "create" in c] and controller.deleted == ["computeinstance-e00new"]
    assert beat["active"] == [{"run_id": "b2-n1", "state": "relaunch_pending"}]
    assert cloud.objects["launch/b2-n1/record.json"]["pending"]["failures"] == 0
    assert not nv.controller_alive(cloud)  # fresh heartbeat, but unfit: this side does not stand back
    cloud.state = "RUNNING"
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "relaunched" and [a["kind"] for a in record["attempts"]] == ["fresh", "resumed"]


def test_a_controller_whose_creates_keep_failing_the_same_way_steps_aside(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    preempt(cloud, "b2-n1", time.time() + 100, {"seq": 1, "elapsed_seconds": 92.66})
    remote = nv.check_config({**CONFIG, "rclone_remote": "arc", "nebius_binary": "/usr/local/bin/arc-nebius"})
    controller = Unschedulable(remote, cloud.objects, instance_state="STOPPED")
    controller.error = ("/usr/local/bin/arc-nebius compute instance create --parent-id exited 7: "
                        "PermissionDenied: compute.instances.create")
    assert nv.controller_loop(remote, controller, rounds=2, interval=0) == "rounds"
    assert cloud.objects["launch/controller/heartbeat.json"]["state"] == "ok" and nv.controller_alive(cloud)
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")  # two failures: still the controller's
    assert info["state"] == "relaunch_pending" and record["pending"]["repeats"] == 2
    assert nv.controller_loop(remote, controller, rounds=3, interval=0) == "rounds"
    beat = cloud.objects["launch/controller/heartbeat.json"]
    assert beat["state"] == "controller_unfit" and beat["unfit"]["reason"] == "create"
    assert "PermissionDenied" in beat["unfit"]["error"] and controller.tries == 3  # no fourth try, no 14th
    cloud.state = "RUNNING"
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1")
    assert info["state"] == "relaunched" and len(record["attempts"]) == 2


def test_the_controller_vm_is_a_small_on_demand_machine_with_no_key_and_its_own_reservation(tmp_path, monkeypatch):
    make_ledger(tmp_path, cap=200)
    config = nv.check_config({**CONFIG, "profile": "arc"})
    cloud = FakeCloud(config, {"runs/validation/controller-1/nebius/status.json": {"phase": "controller"}})
    cloud.objects["launch/controller/heartbeat.json"] = {"epoch": time.time() + 60, "state": "ok",
                                                         "cli_version": "0.12.286", "unfit": None}
    record = nv.controller_up(tmp_path, config, cloud, "controller-1", hours=30, poll=0)
    assert record["controller"] == {"state": "ok", "unfit": None, "cli_version": "0.12.286"}
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["kind"] == "controller" and entry["preemptible"] is False and entry["max_seconds"] == 108000
    assert 0.05 < entry["rate_usd_per_hour"] < 0.10 and entry["charged_bound_usd"] < 4
    create = next(c for c in cloud.calls if "create" in c)
    assert "cpu-d3" in create and "2vcpu-8gb" in create and "--preemptible" not in " ".join(create)
    spec = spec_of(cloud)
    assert spec["mode"] == "controller" and spec["assets"] == [] and record["state"] == "submitted"
    remote = cloud.objects["launch/controller-1/config.json"]  # what the VM's own launcher runs with
    assert remote["profile"] is None and remote["rclone_remote"] == "arc" and remote["cli_version"] == "0.12.286"
    assert remote["nebius_binary"] == "/usr/local/bin/arc-nebius" and nv.check_config(remote)["bucket"] == "arc-bucket"
    assert not [k for k in remote if "secret" in k.lower() and k != "s3_secret_id"]
    uploaded = [Path(c[2]).name for c in cloud.calls if c[:2] == ["rclone", "copyto"]]
    assert uploaded == ["nebius_runner.py", "nebius_validation.py", "config.json"]


SIBLING = """
import modal
HX4_TRANCHE_USD = 200.0
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or ".")


def hx4_study():
    bundle = (Path(os.environ.get("ARC_VALIDATION_BUNDLE", "")).resolve() if modal.is_local()
              else Path("/opt/arc-validation"))
    return json.loads((bundle / "configs/comparison.json").read_text())


def validation_timeout(kind, test):
    if kind == "arena" and test == "hx4-level":
        return min(14400, int(hx4_study()["notebook_seconds"]) + 300)
    return {("arena", "quality-fixed-replicated"): 6600, ("arena", "dev10-preview-1h"): 5400,
            ("arena", "heldout15-preview-1h"): 5700}[(kind, test)]


def validate_preview(study, bundle):
    if study.get("wall_seconds") != 3600:
        raise ValueError("the preview needs a one-hour wall")
    (bundle / "src/arena/quality_eval.py").read_text()


def arena_arguments(test):
    if test == "hx4-level":
        return ["--trial", "level-bench"]
    if test == "heldout15-preview-1h":
        return ["--trial", "duck-transfer"]
    if test in ("quality-fixed-replicated", "dev10-preview-1h"):
        return ["--trial", "duck-transfer"]
    raise ValueError(f"Unsupported arena test: {test}")


@app.function(gpu="x")
def benchmark(run_id: str):
    source = Path("/opt/arc-validation")
    os.environ["A"] = "1"
    if run_id:
        if TEST in ('hx4-level', 'heldout15-preview-1h'):
            raise RuntimeError('a check under another condition belongs to the container')
    if KIND == "arena":
        if TEST == 'heldout15-preview-1h':
            validate_preview(json.loads((source / 'configs/comparison.json').read_text()), source)
        if TEST == 'hx4-level':
            study = json.loads((source / 'configs/comparison.json').read_text())
            if not run_id.startswith('hx4-') or not study.get('level_bench'):
                raise ValueError('hx4-level needs a build_hx4.py bundle and an hx4- run id')
        if TEST in ('quality-fixed-replicated',):
            study = json.loads((source / 'configs/comparison.json').read_text())
            if study.get('replicas') != 4:
                raise ValueError('the quality run needs four replicas')
            if not Path('/assets/games').is_dir():
                raise ValueError('container only')
        if TEST == 'dev10-preview-1h':
            os.environ["B"] = "1"
            raise ValueError('not a guard: it has a side effect')
"""


def test_any_test_the_sibling_modal_launcher_defines_is_planned_from_it(tmp_path):
    path = tmp_path / "modal_validation.py"
    path.write_text(SIBLING)
    bundle = make_bundle(tmp_path, {**STUDY, "notebook_seconds": 5000, "level_bench": {"x": 1}, "trials": [
        *STUDY["trials"], {"name": "level-bench", "kind": "level-bench"}]})
    study = json.loads((bundle / "configs/comparison.json").read_text())
    plan = nv.plan_run("arena", "hx4-level", study, "hx4-a", bundle, path)
    assert plan == {"timeout": 5300, "script": "bootstrap_arena.py", "arguments": ["--trial", "level-bench"],
                    "stop_rule": None, "guards": {"ran": 1, "skipped": []}}
    with pytest.raises(ValueError, match="hx4- run id"):  # the container's own bundle check, run before the launch
        nv.plan_run("arena", "hx4-level", study, "other", bundle, path)
    # A check that reads container paths is reported, not run; a block with side effects is not a check.
    plan = nv.plan_run("arena", "quality-fixed-replicated", study, "q1", bundle, path)
    assert plan["timeout"] == 6600 and plan["guards"]["ran"] == 0 and "container path" in plan["guards"]["skipped"][0]
    assert "guards" not in nv.plan_run("arena", "dev10-preview-1h", study, "d1", bundle, path)
    with pytest.raises(ValueError, match="does not plan it"):
        nv.plan_run("arena", "dev10-preview-long", study, "h1", bundle, path)
    # A check the sibling keeps in a validate_* function runs here too; a bundle without the file it reads, or a
    # test whose trial the bundle does not define, is refused before anything is paid for.
    with pytest.raises(ValueError, match="one-hour wall"):
        nv.plan_run("arena", "heldout15-preview-1h", study, "h1", bundle, path)
    hour = {**study, "wall_seconds": 3600}
    with pytest.raises(ValueError, match="the bundle lacks what this test checks"):
        nv.plan_run("arena", "heldout15-preview-1h", hour, "h1", make_bundle(tmp_path, hour), path)
    resumable = make_bundle(tmp_path, hour, checkpointing=True)
    (resumable / "src/arena/quality_eval.py").write_text("")
    assert nv.plan_run("arena", "heldout15-preview-1h", hour, "h1", resumable, path)["guards"]["ran"] == 1
    with pytest.raises(ValueError, match=r"runs the trial \['level-bench'\] and this bundle's study defines"):
        nv.plan_run("arena", "hx4-level", {**study, "trials": STUDY["trials"]}, "hx4-a", bundle, path)
    assert "ARC_VALIDATION_BUNDLE" not in os.environ and "A" not in os.environ and "B" not in os.environ


def test_mirror_plan_lists_the_missing_overlays(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    config_path = tmp_path / "arc.json"
    config_path.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("ARC_NEBIUS_CONFIG", str(config_path))
    monkeypatch.setenv("ARC_VALIDATION_BUNDLE", str(bundle))
    assets = nv.required_assets(STUDY)
    monkeypatch.setattr(nv, "Cloud", lambda config, dry_run=False: FakeCloud(config, manifests(assets[:-1])))
    nv.main(["mirror-plan"])
    out = capsys.readouterr().out
    assert json.loads(out[:out.index("uv run")])["missing"] == [assets[-1]]
    assert out.strip().endswith(f"--prefixes {assets[-1]}")
    monkeypatch.setattr(nv, "Cloud", lambda config, dry_run=False: FakeCloud(config, manifests(assets)))
    nv.main(["mirror-plan"])
    assert json.loads(capsys.readouterr().out)["missing"] == []


FAKE_BOOTSTRAP = """
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("--bundle", "--working", "--output", "--games-dir", "--max-seconds"):
    p.add_argument(name)
p.add_argument("--trial", action="append")
a = p.parse_args()
root = Path(os.environ["ARC_NEBIUS_ROOT"])
assert (root / "kaggle/input/datasets/o/d/weights.bin").read_bytes() == b"w" * 4096
assert (Path(a.games_dir) / "g1/game.py").is_file() and a.trial == ["duck-transfer"]
assert not [k for k in os.environ if k.startswith(("AWS_", "RCLONE_", "NEBIUS_"))], "credentials leaked"
print("fake bootstrap started", flush=True)
live = Path(a.output) / "live"
live.mkdir(parents=True)
(live / "progress.json").write_text(json.dumps({"live": {"score": {"score": 12.5}}, "levels": 3}))
trials = Path(a.working) / "trials/duck-transfer"
trials.mkdir(parents=True)
(trials / "progress.json").write_text("{}")
stop = Path(os.environ["ARENA_STOP_REQUEST"])
end = time.time() + 60
while not stop.exists() and time.time() < end:
    time.sleep(0.2)
Path(a.output, "bootstrap-results.json").write_text(json.dumps({"duck-transfer": "complete", "stopped": stop.exists()}))
sys.exit(0 if stop.exists() else 3)
"""
# A trial that follows the checkpoint protocol of src/arena/checkpoint.py: generations in its folder, one more when asked.
CHECKPOINT_BOOTSTRAP = """
import argparse, hashlib, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("--bundle", "--working", "--output", "--games-dir", "--max-seconds"):
    p.add_argument(name)
p.add_argument("--trial", action="append")
a = p.parse_args()
assert os.environ["ARC_CHECKPOINT_SECONDS"] == "600" and os.environ["ARC_CHECKPOINT_HOLD"] == "8"
trial = Path(a.working) / "trials/duck-transfer"
folder = trial / "checkpoint"
def generation(seq):
    work = folder / f"gen-{seq:06d}.tmp"
    work.mkdir(parents=True)
    state = b"state %d" % seq
    (work / "state.pkl.z").write_bytes(state)
    (work / "actions.json").write_text("{}")
    (work / "manifest.json").write_text(json.dumps({"seq": seq, "elapsed_seconds": 100.0 * seq, "epoch": time.time(),
                                                    "bytes": len(state), "sha256": hashlib.sha256(state).hexdigest()}))
    work.rename(folder / f"gen-{seq:06d}")
Path(a.output, "live").mkdir(parents=True)
if os.environ.get("ARC_CHECKPOINT_RESUME") == "1":
    assert os.environ["ARC_CHECKPOINT_ATTEMPT"] == "2"
    found = sorted(q.name for q in folder.glob("gen-*"))
    if (folder / "poison").exists():
        (trial / "exception.txt").write_text("arena.checkpoint.ResumeError: environment replay diverged")
        sys.exit(1)
    Path(a.output, "bootstrap-results.json").write_text(json.dumps({"resumed": found, "kept": (trial / "progress.jsonl").read_text()}))
    sys.exit(0)
generation(1)
(trial / "progress.jsonl").write_text("one")
(Path(a.output) / "live/progress.json").write_text("{}")
end = time.time() + 60
while time.time() < end:
    if (folder / "request").exists():
        generation(2)
        (folder / "request").unlink()
    time.sleep(0.1)
"""
needs_rclone = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")


def stage_run(tmp_path, *, corrupt=False, bootstrap=None, machine="vm", more_files=None, **extra):
    """A temp VM root, a folder bucket with assets + manifests, the launch files, and the runner's spec."""
    root, bucket = tmp_path / machine, tmp_path / "s3/arc-bucket"
    files = {"games/g1/game.py": b"# game\n", "flash-input/datasets/o/d/weights.bin": b"w" * 4096,
             **(more_files or {})}
    prefixes = ["games", "flash-input/datasets/o/d"]
    for prefix in prefixes:
        entries = {}
        for rel, body in files.items():
            if rel.startswith(prefix + "/"):
                target = bucket / "assets" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                entries[rel[len(prefix) + 1:]] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        (bucket / "assets-manifests").mkdir(exist_ok=True)
        (bucket / "assets-manifests" / f"{nv.asset_slug(prefix)}.json").write_text(
            json.dumps({"prefix": prefix, "files": entries}))
    if corrupt:
        (bucket / "assets/flash-input/datasets/o/d/weights.bin").write_bytes(b"x" * 4096)
    bundle = make_bundle(tmp_path)
    (bundle / "bootstrap_arena.py").write_text(bootstrap or FAKE_BOOTSTRAP)
    (root / "opt/arc/launch").mkdir(parents=True)
    digest = nv.pack_bundle(bundle, root / "opt/arc/launch/bundle.tar.gz")
    spec = {"run_id": "b2-e2e", "mode": "run", "kind": "arena", "test": "b200-run", "timeout": 600,
            "script": "bootstrap_arena.py", "arguments": ["--trial", "duck-transfer"], "stop_rule": None,
            "gpu": "RTX-PRO-6000", "bucket": "arc-bucket", "runs_prefix": "runs", "assets_prefix": "assets",
            "manifests_prefix": "assets-manifests", "launch_prefix": "launch/b2-e2e", "assets": prefixes,
            "verify": "sha256", "block_network": False, "deadline_epoch": time.time() + 3600,
            "finalize_allowance": 60, "sync_seconds": 1, "stop_poll_seconds": 1, "rate_per_second": 0.0003,
            "bundle_sha256": digest, "instance_id": "computeinstance-test", "image_family": "test", **extra}
    (root / "opt/arc/launch.json").write_text(json.dumps(spec))
    env = {**os.environ, "ARC_NEBIUS_ROOT": str(root), "RCLONE_CONFIG_ARC_TYPE": "alias",
           "RCLONE_CONFIG_ARC_REMOTE": str(tmp_path / "s3"), "AWS_SECRET_ACCESS_KEY": "must-not-leak"}
    process = subprocess.Popen([sys.executable, str(SCRIPTS / "nebius_runner.py"), str(root / "opt/arc/launch.json")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return process, bucket / "runs/validation/b2-e2e", root


def wait_for(path, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.2)
    return False


@needs_rclone
def test_runner_skips_listed_asset_files(tmp_path):
    more = {f"flash-input/datasets/o/d/sub/layer-0000{i}-experts-0000-0127.safetensors": b"e" * 512 for i in range(3)}
    process, remote, root = stage_run(tmp_path, more_files=more,
                                      asset_skip={"flash-input/datasets/o/d": ["sub/layer-*-experts-*.safetensors"]})
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (remote / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    target = root / "assets/flash-input/datasets/o/d"
    assert (target / "weights.bin").stat().st_size == 4096 and not (target / "sub").exists()
    ready = [a for a in json.loads((remote / "execution.json").read_text())["assets"]
             if a["prefix"] == "flash-input/datasets/o/d"][0]
    assert ready["files"] == 1 and ready["bytes"] == 4096
    assert ready["skipped_files"] == 3 and ready["skipped_bytes"] == 1536


def test_asset_skip_rules(tmp_path):
    manifest = {"prefix": "p", "files": {"a.bin": {"size": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
                                         "x/layer-00001-experts-0000-0127.safetensors": {"size": 5, "sha256": "0"}}}
    (tmp_path / "a.bin").write_bytes(b"a")
    skip = runner.skip_list(manifest, ["x/layer-*-experts-*.safetensors"])
    assert skip == ["x/layer-00001-experts-0000-0127.safetensors"]
    result = runner.verify_manifest(tmp_path, manifest, skip=skip)
    assert (result["files"], result["skipped_files"], result["skipped_bytes"]) == (1, 1, 5)
    assert "skipped_files" not in runner.verify_manifest(tmp_path, {"prefix": "p", "files": {
        "a.bin": manifest["files"]["a.bin"]}})
    with pytest.raises(ValueError, match="matches no file"):
        runner.skip_list(manifest, ["nothing-*"])
    assets = ["flash-input/models/a/b"]
    assert nv.asset_skip_for({}, assets) == {}
    with pytest.raises(ValueError, match="does not fetch"):
        nv.asset_skip_for({"nebius_asset_skip": {"flash-input/models/c/d": ["*"]}}, assets)
    with pytest.raises(ValueError, match="non-empty list"):
        nv.asset_skip_for({"nebius_asset_skip": {"flash-input/models/a/b": []}}, assets)


def test_boot_disk_counts_kept_files_and_floor():
    class Cloud:
        dry_run = False

        def read_json(self, prefix, name):
            gib = 2 ** 30
            return {"files": {"keep.safetensors": {"size": 120 * gib},
                              **{f"layer-{i:05d}-experts.safetensors": {"size": gib} for i in range(100)}}}

    config = {"boot_disk_gib": 372, "boot_disk_type": "network_ssd_non_replicated"}
    assets = ["flash-input/models/a/b"]
    assert nv.boot_disk_for(config, Cloud(), assets) == 372  # (38 + 220 + 20) x 1.15 = 320
    skip = {"flash-input/models/a/b": ["layer-*-experts.safetensors"]}
    assert nv.boot_disk_for(config, Cloud(), assets, skip) == 279  # (38 + 120 + 20) x 1.15 = 205
    assert nv.boot_disk_for(config, Cloud(), assets, skip, 1116) == 1116
    assert nv.boot_disk_for(config, Cloud(), assets, None, 93) == 372  # a floor never shrinks the need
    with pytest.raises(ValueError, match="multiple of 93"):
        nv.boot_disk_for(config, Cloud(), assets, skip, 1000)


@needs_rclone
def test_runner_end_to_end_with_stop_request(tmp_path):
    process, remote, root = stage_run(tmp_path)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        assert json.loads((remote / "nebius/status.json").read_text())["phase"] == "bootstrap"
        (remote / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))  # `stop RUN_ID`
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "complete" and final["exit_code"] == 0 and final["synced"] is True
    assert json.loads((remote / "arena/bootstrap-results.json").read_text())["stopped"] is True
    assert "fake bootstrap started" in (remote / "bootstrap.log").read_text()
    execution = json.loads((remote / "execution.json").read_text())
    assert execution["platform"] == "nebius" and execution["exit_code"] == 0
    assert [a["prefix"] for a in execution["assets"]] == ["games", "flash-input/datasets/o/d"]
    assert execution["command"][1:6] == [str(root / "opt/arc-validation/bootstrap_arena.py"), "--bundle",
                                         str(root / "opt/arc-validation"), "--working", str(root / "tmp/arena")]
    assert os.readlink(root / "kaggle/input") == str(root / "assets/flash-input")
    phases = [h["phase"] for h in json.loads((remote / "nebius/status.json").read_text())["history"]]
    assert phases == ["boot", "assets", "bootstrap"]


@needs_rclone
def test_runner_marks_an_interruption_and_keeps_the_working_files(tmp_path):
    process, remote, root = stage_run(tmp_path)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        process.send_signal(signal.SIGTERM)  # the preemption notice
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 1, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "interrupted" and final["interrupted"]["signal"] == "SIGTERM"
    assert final["interrupted"]["phase"] == "bootstrap" and final["synced"] is True
    assert (remote / "arena/working/trials/duck-transfer/progress.json").is_file()
    assert (remote / "bootstrap.log").is_file() and (remote / "execution.json").is_file()
    assert nv.classify(None, {"phase": "bootstrap"}, final, None, time.time(), time.time() + 60) == "preempted"


@needs_rclone
def test_runner_refuses_assets_that_differ_from_the_manifest(tmp_path):
    process, remote, root = stage_run(tmp_path, corrupt=True)
    output = process.communicate(timeout=90)[0]
    assert process.returncode == 1, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and "sha256 mismatch" in final["error"]
    assert not (root / "opt/arc-validation").exists() and not (remote / "bootstrap.log").exists()


@needs_rclone
def test_runner_uploads_checkpoints_takes_one_at_the_notice_and_a_relaunch_restores_them(tmp_path):
    staged = {"bootstrap": CHECKPOINT_BOOTSTRAP, "checkpoint": {"seconds": 600}}
    process, remote, root = stage_run(tmp_path, attempt=1, **staged)
    kept = remote / "arena/working/trials/duck-transfer/checkpoint"
    try:
        assert wait_for(remote / "nebius/checkpoint.json"), process.communicate(timeout=5)[0]
        assert wait_for(remote / "arena/live/progress.json")
        assert (kept / "gen-000001/state.pkl.z").read_bytes() == b"state 1"  # in the bucket while the run goes on
        process.send_signal(signal.SIGTERM)
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "interrupted" and final["resume_failed"] is False, output
    assert final["final_checkpoint"]["written"] == ["duck-transfer"] and final["final_checkpoint"]["seconds"] < 5
    assert final["checkpoint"]["duck-transfer"]["seq"] == 2 and final["checkpoint"]["duck-transfer"]["elapsed_seconds"] == 200
    marker = json.loads((remote / "nebius/checkpoint.json").read_text())
    assert marker["trials"]["duck-transfer"]["seq"] == 2 and marker["done"] == [] and marker["attempt"] == 1
    assert sorted(q.name for q in kept.iterdir()) == ["gen-000001", "gen-000002"]  # no request file, no .tmp
    record = {"policy": {"auto_relaunch": True, "max_attempts": 4, "checkpointing": True}, "attempts": [{}]}
    assert nv.decide(record, nv.classify("STOPPED", None, final, None, time.time(), time.time() + 9), final,
                     marker) == "resume"

    # The next machine: an empty disk, the same output folder.
    process, remote, second = stage_run(tmp_path, machine="vm2", attempt=2, **staged)
    output = process.communicate(timeout=90)[0]
    assert process.returncode == 0, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "complete" and final["attempt"] == 2 and final["restored"]["duck-transfer"]["seq"] == 2
    results = json.loads((remote / "arena/bootstrap-results.json").read_text())
    assert results == {"resumed": ["gen-000001", "gen-000002"], "kept": "one"}

    # A checkpoint the trial cannot continue is reported as such (the launcher then starts from zero).
    (kept / "poison").write_text("x")
    process, remote, third = stage_run(tmp_path, machine="vm3", attempt=2, **staged)
    output = process.communicate(timeout=90)[0]
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and final["resume_failed"] is True, output
    assert nv.decide({**record, "attempts": [{}, {}]}, "failed", final, marker) == "fresh"


def test_a_launch_is_a_fixed_instance_with_one_attempt_and_no_relaunch_machinery(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    objects = {**manifests(nv.required_assets(STUDY)),
               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}}
    cloud = FakeCloud(config, objects)
    record = nv.launch(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0,
                       approval=APPROVED, bundle=make_bundle(tmp_path, checkpointing=True))
    create = next(c for c in cloud.calls if "create" in c)
    assert "--on-demand" in create and "--preemptible-on-preemption" not in create
    assert "--spot-pricing-policy-id" not in create
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    seconds = 86400 + 2400 + 300  # one attempt: no relaunch reservation
    assert entry["preemptible"] is False and entry["max_seconds"] == seconds and entry["max_attempts"] == 1
    assert entry["gpu_rate_usd_per_hour"] == 1.80 and entry["rate_basis"] == "on-demand list price"
    assert entry["charged_bound_usd"] == pytest.approx(nv.bound_usd(seconds, 1.80 + 744 * 0.053 / 730))
    policy = record["policy"]
    assert policy["auto_relaunch"] is False and policy["checkpointing"] is False and policy["max_attempts"] == 1
    assert spec_of(cloud)["checkpoint"] is None  # a bundle that could checkpoint does not, unless asked
    # A fixed VM that stops without a final marker is not relaunched: the run ends, the ledger is settled.
    cloud.state = "STOPPED"
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=5000.0)
    assert record["state"] == "stopped" and len(record["attempts"]) == 1 and record.get("pending") is None
    assert json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]["state"] == "stopped"
    # The controller leaves fixed runs alone.
    objects["launch/b2-n1/record.json"]["state"] = "submitted"
    controller = FakeCloud(nv.check_config({**CONFIG, "rclone_remote": "arc"}), objects)
    nv.controller_loop(controller.config, controller, rounds=1)
    assert objects["launch/controller/heartbeat.json"]["active"] == []

    # Checkpoints on a fixed instance are an option: written, not resumed from by anything.
    cloud = FakeCloud(config, {**objects, "runs/validation/b2-n2/nebius/status.json": {"phase": "bootstrap"}})
    record = nv.launch(tmp_path, config, cloud, run_id="b2-n2", kind="arena", test="b200-run", approval=APPROVED,
                       bundle=make_bundle(tmp_path, checkpointing=True), checkpoint=True)
    assert spec_of(cloud)["checkpoint"] == {"seconds": 600} and record["policy"]["auto_relaunch"] is False
    assert record["max_seconds"] == seconds


def test_a_fixed_launch_without_capacity_or_quota_fails_cleanly_at_no_cost(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = Unschedulable(config, manifests(nv.required_assets(STUDY)))
    with pytest.raises(RuntimeError, match=r"b2-n1 was not started \(no capacity\): no VM or disk is left"):
        nv.launch(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "launch_failed_no_execution" and entry["charged_bound_usd"] == 0.0
    assert cloud.deleted == ["computeinstance-unscheduled"] and "no capacity" in entry["note"]
    assert nv.read_record(tmp_path, "b2-n1")["reason"] == "no capacity"
    # The fourth VM, 7 October: the refusal carries the provider's violation, not the request ids after it.
    stderr = ("Error: Quota failure QuotaFailure: service VPC API, violations: vpc.ipv4-address.public.count "
              "(limit 3, requested 4)\n  request id: 5f0c2f6e-93f4-4b53-8f2c-0a1d2e3f4a5b\n"
              "  trace id: 9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d\n" + "  caused by: rpc error\n" * 3)
    said = nv.explain(stderr)
    assert said.startswith("Error: Quota failure QuotaFailure: service VPC API, violations: "
                           "vpc.ipv4-address.public.count (limit 3, requested 4)")
    assert "request id" not in said and "5f0c2f6e" not in said and "9a8b7c6d" not in said
    assert nv.explain(stderr) == nv.explain(stderr.replace("5f0c2f6e", "11111111"))  # the same failure, twice
    assert nv.explain("something odd\nrequest id: abc\nwent away") == "something odd | went away"
    cloud.error = f"nebius --profile arc compute instance create exited 1: {said}"
    with pytest.raises(RuntimeError) as refused:
        nv.launch(tmp_path, config, cloud, run_id="b2-n2", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    message = str(refused.value)
    assert "b2-n2 was not started (quota exceeded)" in message and "public_ip false" in message
    assert "Nebius: Error: Quota failure QuotaFailure: service VPC API, violations: vpc.ipv4-address.public.count " \
           "(limit 3, requested 4)" in message
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][-1]
    assert entry["charged_bound_usd"] == 0.0 and "limit 3, requested 4" in entry["note"]


def test_a_launch_refuses_when_the_quota_or_our_own_ceiling_has_no_room(tmp_path):
    make_ledger(tmp_path, cap=2000)
    vms = [{"metadata": {"id": f"computeinstance-{i}", "name": f"arc-r{i}", "labels": {"arc-run": f"r{i}"}},
            "status": {"state": "RUNNING"}} for i in range(2)]
    vms.append({"metadata": {"id": "computeinstance-c", "name": "arc-controller",
                             "labels": {"arc-run": "c", "arc-test": "controller"}}, "status": {"state": "RUNNING"}})
    allowances = [
        {"metadata": {"name": "compute.instance.gpu.rtx6000"}, "spec": {"limit": "2", "region": "eu-south1"},
         "status": {"usage": "2"}},
        {"metadata": {"name": "compute.instance.gpu.rtx6000.preemptible"},
         "spec": {"limit": "8", "region": "eu-south1"}, "status": {"usage": "1"}},
        {"metadata": {"name": "compute.instance.gpu.rtx6000"}, "spec": {"limit": "9", "region": "us-central1"},
         "status": {"usage": "0"}},
        {"metadata": {"name": "compute.instance.gpu.h200"}, "spec": {"limit": "0", "region": "eu-south1"}},
    ]

    class Busy(FakeCloud):
        def run(self, argv, **kwargs):
            done = super().run(argv, **kwargs)
            if "quotas" in argv:
                done.stdout = json.dumps({"items": allowances})
            elif "list" in argv and "compute" in argv:
                done.stdout = json.dumps({"items": vms})
            return done

    config = nv.check_config(CONFIG)
    cloud = Busy(config, manifests(nv.required_assets(STUDY)))
    assert cloud.gpu_quota(False) == {"name": "compute.instance.gpu.rtx6000", "limit": 2, "usage": 2}
    assert cloud.gpu_quota(True)["limit"] == 8 and len(cloud.gpu_vms()) == 2
    with pytest.raises(RuntimeError, match="Quota: compute.instance.gpu.rtx6000 allows 2 and 2 are in use"):
        nv.launch(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    assert json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"] == []  # refused before reserving
    assert not [c for c in cloud.calls if "create" in c]
    allowances[0]["status"]["usage"] = "1"
    capped = nv.check_config({**CONFIG, "max_parallel_gpu_vms": 2})
    with pytest.raises(RuntimeError, match="2 GPU VMs of ours exist and max_parallel_gpu_vms is 2"):
        nv.launch(tmp_path, capped, Busy(capped, cloud.objects), run_id="b2-n1", kind="arena", test="b200-run",
                  approval=APPROVED, bundle=make_bundle(tmp_path))
    # An allowance list this code cannot read decides nothing.
    assert FakeCloud(config).gpu_quota(False) is None
    # Every VM takes a public IPv4 address: none free is refused before anything is reserved or created.
    allowances.append({"metadata": {"name": "vpc.ipv4-address.public.count"},
                       "spec": {"limit": "3", "region": "eu-south1"}, "status": {"usage": "3"}})
    busy = Busy(config, cloud.objects)
    before = (tmp_path / "docs/spend.json").read_text()
    with pytest.raises(RuntimeError, match="No public IPv4 address is free: vpc.ipv4-address.public.count allows 3 "
                                           "and 3 are in use"):
        nv.launch(tmp_path, config, busy, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    assert (tmp_path / "docs/spend.json").read_text() == before and not [c for c in busy.calls if "create" in c]
    del allowances[-1]["status"]["usage"]  # usage not reported: our own VMs (2 GPU + the controller) hold 3
    with pytest.raises(RuntimeError, match="allows 3 and 3 are in use"):
        nv.launch(tmp_path, config, busy, run_id="b2-n1", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    # Without public addresses (config public_ip false) the address quota is not this launch's business.
    private = nv.check_config({**CONFIG, "public_ip": False})
    cloud = Busy(private, {**cloud.objects, "runs/validation/b2-n4/nebius/status.json": {"phase": "boot"}})
    nv.launch(tmp_path, private, cloud, run_id="b2-n4", kind="arena", test="b200-run", approval=APPROVED,
              bundle=make_bundle(tmp_path))
    create = next(c for c in cloud.calls if "create" in c)
    assert json.loads(create[create.index("--network-interfaces") + 1]) == [
        {"name": "eth0", "ip_address": {}, "subnet_id": CONFIG["subnet_id"]}]
    allowances.pop()
    # The project's real entry, 7 October: no limit and no usage. Unknown, so the create decides (it succeeded).
    allowances[:] = [
        {"metadata": {"name": "compute.instance.gpu.rtx6000"}, "spec": {"region": "eu-south1"},
         "status": {"service": "compute", "description": "Total number of NVIDIA RTX PRO 6000 GPUs across all VMs",
                    "unit": "count", "usage_state": "USAGE_STATE_NOT_USED"}},
        {"metadata": {"name": "compute.instance.preemptible.count"}, "spec": {"region": "eu-south1"},
         "status": {"service": "compute", "unit": "count", "usage_state": "USAGE_STATE_NOT_USED"}},
        {"metadata": {"name": "compute.instance.count"}, "spec": {"region": "eu-south1"},
         "status": {"service": "compute", "unit": "count", "usage_state": "USAGE_STATE_NOT_USED"}},
    ]
    vms[:] = []
    cloud = Busy(config, {**cloud.objects, "runs/validation/b2-n3/nebius/status.json": {"phase": "boot"}})
    assert cloud.gpu_quota(False) is None and cloud.gpu_quota(True) is None
    record = nv.launch(tmp_path, config, cloud, run_id="b2-n3", kind="arena", test="b200-run", approval=APPROVED,
                       bundle=make_bundle(tmp_path))
    assert record["state"] == "submitted" and len([c for c in cloud.calls if "create" in c]) == 1


def test_stop_without_a_reason_is_a_message_not_a_traceback(tmp_path, monkeypatch):
    config_path = tmp_path / "arc.json"
    config_path.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("ARC_NEBIUS_CONFIG", str(config_path))
    monkeypatch.setenv("ARC_VALIDATION_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match='stop needs --reason "<why>"'):
        nv.main(["stop", "b2-n1", "--yes"])
    done = subprocess.run([sys.executable, str(SCRIPTS / "nebius_validation.py"), "status", "no-such-run", "--dry-run"],
                          capture_output=True, text=True, env={**os.environ, "ARC_NEBIUS_CONFIG": str(config_path),
                                                               "ARC_VALIDATION_ROOT": str(tmp_path)})
    assert done.returncode == 1 and "Traceback" not in done.stderr and done.stderr.startswith("error: ")


SCHEDULE_TIMEOUT = ("compute instance create exited 65: Error: VM schedule timeout, most likely due to insufficient "
                    "hardware resources (requested 1gpu-24vcpu-218gb)")  # the first smoke launch, 7 October


class Unschedulable(FakeCloud):
    """Creates fail the way Nebius reports no capacity; the unscheduled instance stays behind, STOPPED."""

    error, capacity = SCHEDULE_TIMEOUT, False

    def run(self, argv, **kwargs):
        if "--help" in argv:
            return super().run(argv, **kwargs)
        self.tries = getattr(self, "tries", 0) + ("create" in argv)
        if "create" in argv and not self.capacity:
            self.left = [{"metadata": {"id": "computeinstance-unscheduled", "name": argv[argv.index("--name") + 1],
                                       "labels": {"arc-run": "x"}}, "status": {"state": "STOPPED"}}]
            self.create_timeout = kwargs.get("timeout")
            raise RuntimeError(self.error)
        done = super().run(argv, **kwargs)
        if "list" in argv:
            done.stdout = json.dumps({"items": getattr(self, "left", [])})
        elif "delete" in argv:
            self.left = []
        return done


def test_failed_create_releases_the_reservation(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = Unschedulable(config, manifests(nv.required_assets(STUDY)))
    cloud.error = "compute instance create exited 7: PermissionDenied: compute.instances.create"
    with pytest.raises(RuntimeError, match="PermissionDenied"):
        launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=make_bundle(tmp_path),
                  approval=APPROVED)
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "launch_failed_no_execution" and entry["charged_bound_usd"] == 0.0
    assert entry["previous_charged_bound_usd"] > 30 and cloud.deleted == ["computeinstance-unscheduled"]
    assert nv.read_record(tmp_path, "b2-n1")["state"] == "launch_failed_no_execution"


def test_no_capacity_on_the_first_create_leaves_nothing_behind_and_the_run_pending(tmp_path, capsys):
    for text in (SCHEDULE_TIMEOUT, "ResourceExhausted: not enough resources", "no available capacity in eu-south1"):
        assert nv.NO_CAPACITY.search(text)
    for text in ("PermissionDenied: compute.instances.create", "quota exceeded for gpu", "AlreadyExists: arc-b2-n1"):
        assert not nv.NO_CAPACITY.search(text)
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "capacity_retry_seconds": 1800})
    objects = manifests(nv.required_assets(STUDY))
    cloud = Unschedulable(config, objects)
    record = launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0,
                       approval=APPROVED, bundle=make_bundle(tmp_path, checkpointing=True))
    # No traceback, a receipt here and in the bucket, the unscheduled instance (and with it its boot disk) deleted.
    assert record["state"] == "relaunch_pending" and record["attempts"] == [] and cloud.create_timeout == 900
    assert record["pending"]["reason"] == "no_capacity" and "schedule timeout" in record["pending"]["last_error"]
    assert cloud.deleted == ["computeinstance-unscheduled"] and cloud.left == []
    assert nv.read_record(tmp_path, "b2-n1")["state"] == "relaunch_pending"
    assert objects["launch/b2-n1/record.json"]["pending"]["kind"] == "fresh"
    assert '"event": "no_capacity"' in capsys.readouterr().out
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "reserved"  # the run has not begun; nothing is settled, nothing is "lost"
    info, record = nv.inspect(tmp_path, cloud, "b2-n1", now=1100.0)  # `status` leaves it pending
    assert info["state"] == "relaunch_pending" and record["state"] == "relaunch_pending"

    # The controller retries from the bucket record; while its create is in flight this side keeps off.
    remote = nv.check_config({**CONFIG, "capacity_retry_seconds": 1800, "rclone_remote": "arc"})
    controller = Unschedulable(remote, objects)
    info, record = nv.advance(None, remote, controller, "b2-n1", now=1500.0)
    assert record["pending"]["failures"] == 2 and controller.deleted == ["computeinstance-unscheduled"]
    assert objects["launch/b2-n1/claim.json"]["by"] is None  # released once the create has answered
    objects["launch/b2-n1/claim.json"] = {"by": "controller", "epoch": 1550.0}  # its next create, in flight
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=1600.0)
    assert info["claimed_by"] == "controller" and cloud.tries == 1
    assert record["pending"]["failures"] == 2  # the controller's count is the receipt's
    objects["launch/b2-n1/claim.json"] = {"by": None}

    # Capacity returns: attempt 1 of the same run id, and only its seconds are charged.
    controller.capacity = True
    objects["runs/validation/b2-n1/nebius/status.json"] = {"phase": "bootstrap"}
    info, record = nv.advance(None, remote, controller, "b2-n1", now=2000.0)
    assert info["state"] == "relaunched" and [(a["n"], a["run_id"], a["kind"]) for a in record["attempts"]] == [
        (1, "b2-n1", "fresh")]
    assert spec_of(controller)["attempt"] == 1 and record["created_epoch"] == 2000.0
    controller.state = None
    objects["runs/validation/b2-n1/nebius/final.json"] = {"state": "complete", "ended_epoch": 5000.0}
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=5100.0)
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "complete" and entry["observed_seconds"] == 3000


def test_a_run_that_never_got_a_vm_ends_as_relaunch_failed_and_costs_nothing(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config({**CONFIG, "capacity_retry_seconds": 1800})
    cloud = Unschedulable(config, manifests(nv.required_assets(STUDY)))
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0, approval=APPROVED,
              bundle=make_bundle(tmp_path))
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=2000.0)
    assert record["state"] == "relaunch_pending" and record["pending"]["failures"] == 2
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=2900.0)
    assert record["state"] == "relaunch_failed" and record["attempts"] == [] and cloud.left == []
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert entry["state"] == "relaunch_failed" and entry["charged_bound_usd"] == 0.0
    # Ended by the controller instead: this side settles the same way when it next looks.
    make_ledger(tmp_path, cap=2000)
    remote = Unschedulable(config, cloud.objects)
    launch_spot(tmp_path, config, cloud, run_id="b2-n2", kind="arena", test="b200-run", now=1000.0, approval=APPROVED,
              bundle=make_bundle(tmp_path))
    info, record = nv.advance(None, config, remote, "b2-n2", now=2900.0)
    assert record["state"] == "relaunch_failed" and nv.read_record(tmp_path, "b2-n2")["state"] == "relaunch_pending"
    info, record = nv.inspect(tmp_path, cloud, "b2-n2", now=3000.0)
    entry = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]
    assert record["state"] == "relaunch_failed" and entry["state"] == "relaunch_failed"
    assert entry["charged_bound_usd"] == 0.0


def test_a_launcher_that_died_in_the_create_is_picked_up_not_lost(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)

    class Killed(Unschedulable):
        def run(self, argv, **kwargs):
            if "create" in argv and not self.capacity:
                super().run(["list"], **kwargs)
                self.left = [{"metadata": {"id": "computeinstance-orphan", "name": "arc-b2-n1",
                                           "labels": {"arc-run": "b2-n1"}}, "status": {"state": "STOPPED"}}]
                raise KeyboardInterrupt  # the laptop went away mid-call
            return super().run(argv, **kwargs)

    cloud = Killed(config, {**manifests(nv.required_assets(STUDY)),
                            "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    with pytest.raises(KeyboardInterrupt):
        launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0,
                  approval=APPROVED, bundle=make_bundle(tmp_path))
    info, record = nv.inspect(tmp_path, cloud, "b2-n1", now=1300.0)  # the create may still be answering
    assert info["state"] == "scheduling" and cloud.deleted == []
    info, record = nv.inspect(tmp_path, cloud, "b2-n1", now=2000.0)
    assert info["state"] == "relaunch_pending" and cloud.deleted == ["computeinstance-orphan"]
    assert json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]["state"] == "reserved"
    cloud.capacity = True
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=2100.0)
    assert info["state"] == "relaunched" and record["attempts"][0]["run_id"] == "b2-n1"


def test_a_relaunched_vm_without_a_heartbeat_is_deleted_and_ends_the_run(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = FakeCloud(config, {**manifests(nv.required_assets(STUDY)),
                               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}})
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0, approval=APPROVED,
              bundle=make_bundle(tmp_path, checkpointing=True))
    preempt(cloud, "b2-n1", 5000.0, {"seq": 4, "elapsed_seconds": 2400.0})
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=5100.0)
    assert info["state"] == "relaunched"
    cloud.state = "RUNNING"  # the second VM is up, but its cloud-init never reports
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=5100.0 + 700)
    assert info["state"] == "starting" and record["state"] == "submitted"
    info, record = nv.advance(tmp_path, config, cloud, "b2-n1", now=5100.0 + 900)
    assert info["state"] == "failed" and record["state"] == "failed" and len(record["attempts"]) == 2
    assert cloud.deleted[-1] == "computeinstance-e00new2" and "no heartbeat" in record["final"]["error"]


def test_reap_lists_then_deletes_stopped_and_overdue_vms(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    objects = {**manifests(nv.required_assets(STUDY)), "runs/validation/b2-n1/nebius/status.json": {"phase": "boot"}}
    cloud = FakeCloud(config, objects)
    launch_spot(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", bundle=make_bundle(tmp_path),
              approval=APPROVED)
    listed = [
        {"metadata": {"id": "computeinstance-e00new", "name": "arc-b2-n1",
                      "labels": {"arc-run": "b2-n1", "arc-deadline": str(int(time.time()) + 999)}},
         "status": {"state": "STOPPED"}},
        {"metadata": {"id": "computeinstance-old", "name": "arc-x", "labels": {"arc-run": "x", "arc-deadline": "5"}},
         "status": {"state": "RUNNING"}},
        {"metadata": {"id": "computeinstance-ok", "name": "arc-y",
                      "labels": {"arc-run": "y", "arc-deadline": str(int(time.time()) + 999)}},
         "status": {"state": "RUNNING"}},
        {"metadata": {"id": "computeinstance-other", "name": "unrelated", "labels": {}}, "status": {"state": "RUNNING"}},
    ]
    original = cloud.run

    def run(argv, **kwargs):
        done = original(argv, **kwargs)
        if "list" in argv:
            done.stdout = json.dumps({"items": listed})
        return done

    cloud.run = run
    cloud.state = "STOPPED"
    rows = nv.reap(tmp_path, cloud)
    assert [bool(r["reap"]) for r in rows] == [True, True, False] and cloud.deleted == []
    rows = nv.reap(tmp_path, cloud, yes=True)
    assert cloud.deleted == ["computeinstance-e00new", "computeinstance-old"]
    # Stopped by the platform without a final marker: preempted, so the run goes on (from zero: no checkpoint).
    record = nv.read_record(tmp_path, "b2-n1")
    assert rows[0]["handled"] == "relaunch_pending" and record["pending"]["kind"] == "fresh"
    assert record["attempts"][0]["state"] == "preempted"
    assert json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"][0]["state"] == "reserved"


def test_boot_disk_grows_with_assets():
    class Cloud:
        dry_run = False

        def __init__(self, sizes):
            self.sizes = sizes

        def read_json(self, prefix, name):
            return {"files": {f"f{i}": {"size": s} for i, s in enumerate(self.sizes[name])}}

    config = {"boot_disk_gib": 372, "boot_disk_type": "network_ssd_non_replicated"}
    gib = 2 ** 30
    small = Cloud({nv.asset_slug("flash-input/models/a/b") + ".json": [126 * gib, 40 * gib]})
    # 17 Oct: (38 image + assets + 20 output) x 1.15, whole 93 GiB steps, no 372 floor.
    assert nv.boot_disk_for(config, small, ["flash-input/models/a/b"]) == 279  # 166 GiB: DKb-like
    big = Cloud({nv.asset_slug("flash-input/models/a/b") + ".json": [126 * gib, 47 * gib],
                 nv.asset_slug("flash-input/models/c/d") + ".json": [169 * gib]})
    assert nv.boot_disk_for(config, big, ["flash-input/models/a/b", "flash-input/models/c/d"]) == 465  # 342: AV2C
    mid = Cloud({nv.asset_slug("flash-input/models/a/b") + ".json": [203 * gib]})
    assert nv.boot_disk_for(config, mid, ["flash-input/models/a/b"]) == 372  # arc2-like


class StuckStarting(FakeCloud):
    """14 Oct: the create times out and the instance stays STARTING; its delete is accepted but does not finish."""

    def run(self, argv, **kwargs):
        if "--help" in argv:
            return super().run(argv, **kwargs)
        if "create" in argv:
            self.name = argv[argv.index("--name") + 1]
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or 900)
        done = super().run(argv, **kwargs)
        if "list" in argv:
            done.stdout = json.dumps({"items": [{"metadata": {
                "id": "computeinstance-stuck", "name": getattr(self, "name", "x"),
                "labels": {"arc-run": "b2-t1", "arc-deadline": str(int(time.time()) + 3600)}},
                "status": {"state": "STARTING"}}] if hasattr(self, "name") else []})
        return done


def test_a_create_timeout_deletes_the_vm_marks_it_cancelled_and_reap_retries(tmp_path):
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    cloud = StuckStarting(config, manifests(nv.required_assets(STUDY)), instance_state="STARTING")
    with pytest.raises(RuntimeError, match=r"b2-t1 was not started \(no capacity\)"):
        nv.launch(tmp_path, config, cloud, run_id="b2-t1", kind="arena", test="b200-run", approval=APPROVED,
                  bundle=make_bundle(tmp_path))
    assert cloud.deleted == ["computeinstance-stuck"]
    marker = cloud.objects["runs/validation/b2-t1/" + nv.CANCEL_MARKER]
    assert marker["launch_attempt"] == 1 and marker["instance_name"] == cloud.name
    # The marker is written before the delete is asked for.
    order = [i for i, c in enumerate(cloud.calls) if (c[:2] == ["rclone", "copyto"] and "cancelled.json" in c[3])
             or "delete" in c]
    assert "cancelled.json" in cloud.calls[order[0]][3]
    record = nv.read_record(tmp_path, "b2-t1")
    assert record["state"] == "launch_failed_no_execution" and record["deleted_instances"] == ["computeinstance-stuck"]
    rows = nv.reap(tmp_path, cloud)
    assert rows[0]["reap"].startswith("its launch failed") and cloud.deleted == ["computeinstance-stuck"]
    nv.reap(tmp_path, cloud, yes=True)
    assert cloud.deleted == ["computeinstance-stuck", "computeinstance-stuck"]


def test_the_runner_copies_exchange_in_at_boot_and_never_twice_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    spec = {"run_id": "b2-t1", "bucket": "arc-bucket", "runs_prefix": "runs", "kind": "arena", "test": "b200-run",
            "gpu": "RTX-PRO-6000", "timeout": 600, "launch_attempt": 1, "instance_id": "computeinstance-x",
            "deadline_epoch": time.time() + 600, "finalize_allowance": 120, "sync_seconds": 3600,
            "exchange_in": ["b2-other/boot-cache"]}
    copies, gate, started = [], threading.Event(), threading.Event()

    def rclone(self, *args, timeout=None, check=True):
        if args[0] == "copy" and "/exchange-in/" in args[2]:
            copies.append((args[1], time.time()))
            started.set()
            gate.wait(5)
        return subprocess.CompletedProcess(args, 3 if args[0] == "cat" else 0, "", "")

    def fetch_assets(self):
        assert started.wait(5), "exchange-in must start before the assets finish"
        assert self.exchange_in() is False  # a sync pass while the boot copy runs is skipped
        gate.set()
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner.Runner, "rclone", rclone)
    monkeypatch.setattr(runner.Runner, "fetch_assets", fetch_assets)
    monkeypatch.setattr(runner.Runner, "write_diag", lambda self: None)
    runner.Runner(spec).execute()
    assert [c[0] for c in copies] == ["arc:arc-bucket/runs/validation/b2-other/boot-cache"]


def test_the_runner_refuses_to_start_under_the_launchers_cancel_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    spec = {"run_id": "b2-t1", "bucket": "arc-bucket", "runs_prefix": "runs", "kind": "arena", "test": "b200-run",
            "gpu": "RTX-PRO-6000", "timeout": 600, "launch_attempt": 1, "instance_id": "computeinstance-stuck",
            "deadline_epoch": time.time() + 600, "finalize_allowance": 120}
    put, marker = [], {"launch_attempt": 1, "instance_name": "arc-b2-t1"}

    def rclone(self, *args, timeout=None, check=True):
        if args[0] == "cat":
            out = json.dumps(marker) if args[1].endswith("/nebius/cancelled.json") and marker else ""
            return subprocess.CompletedProcess(args, 0 if out else 3, out, "")
        put.append(args[2] if args[0] == "copyto" else args[0])
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runner.Runner, "rclone", rclone)
    monkeypatch.setattr(runner.Runner, "fetch_assets", lambda self: pytest.fail("a cancelled launch must not run"))
    assert runner.Runner(spec).execute() == 1
    assert put == ["arc:arc-bucket/runs/validation/b2-t1/nebius/cancelled-boot.json"]  # no final.json, no status
    # A marker from an earlier launch attempt does not stop a later one that shares the folder.
    marker["launch_attempt"] = 0
    assert runner.Runner(spec).launch_cancelled() is None


# -- baked boot images (opt-in, 21 Oct) ----------------------------------------------------------------------------
def test_bake_spec_is_opt_in():
    assert nv.bake_spec(STUDY) == {}
    assert nv.bake_spec({**STUDY, "nebius_bake": True}) == {"mode": "bake"}
    assert nv.bake_spec({**STUDY, "nebius_image_id": "computeimage-e00x"}) == {
        "image_id": "computeimage-e00x", "use_baked": True}
    with pytest.raises(ValueError, match="together"):
        nv.bake_spec({**STUDY, "nebius_bake": True, "nebius_image_id": "computeimage-e00x"})
    with pytest.raises(ValueError, match="computeimage-"):
        nv.bake_spec({**STUDY, "nebius_image_id": "ubuntu"})


def test_create_command_boots_a_baked_image_only_when_asked():
    config = nv.check_config(CONFIG)
    stock = nv.create_command(config, "n", {}, "UD")
    assert "--boot-disk-managed-disk-source-image-family-image-family" in stock
    assert "--boot-disk-managed-disk-source-image-id" not in stock
    baked = nv.create_command(config, "n", {}, "UD", image_id="computeimage-e00x")
    i = baked.index("--boot-disk-managed-disk-source-image-id")
    assert baked[i + 1] == "computeimage-e00x"
    assert "--boot-disk-managed-disk-source-image-family-image-family" not in baked


def test_self_delete_powers_off_a_bake_only():
    script = nv.render_bootstrap(spec_for(), nv.check_config(CONFIG))
    guard = "[ -f /run/arc/bake-keep ] && { rm -f /var/log/arc/*.log; : > /var/log/cloud-init-output.log; sync; shutdown -h now; exit 0; }"
    assert guard in script
    assert script.index(guard) < script.index("compute instance delete")


def test_reap_keeps_a_finished_bake_for_its_image():
    deadline = 1_000_000
    labels = {"arc-deadline": str(deadline)}
    baked = {"state": "baked"}
    assert nv.reap_reason(labels, "STOPPED", baked, deadline + 60) is None
    assert nv.reap_reason(labels, "STOPPED", baked, deadline + nv.BAKE_KEEP_SECONDS + 1)
    assert nv.reap_reason(labels, "STOPPED", {"state": "complete"}, deadline - 60)
    assert nv.reap_reason(labels, "STOPPED", None, deadline - 60)
    assert "baked" in nv.TERMINAL


@needs_rclone
def test_runner_bakes_then_a_baked_boot_skips_the_fetch(tmp_path):
    process, remote, root = stage_run(tmp_path, mode="bake")
    output = process.communicate(timeout=120)[0]
    assert process.returncode == 0, output
    assert json.loads((remote / "nebius/final.json").read_text())["state"] == "baked"
    record = json.loads((root / "opt/arc/baked-assets.json").read_text())
    assert set(record) == {"games", "flash-input/datasets/o/d"}
    assert (root / "run/arc/bake-keep").is_file()
    assert not (root / "opt/arc-validation").exists()  # a bake never unpacks the bundle into the image
    # a VM from that image: the disk's assets and record, nothing else
    vm2 = tmp_path / "vm2"
    shutil.copytree(root / "assets", vm2 / "assets")
    (vm2 / "opt/arc").mkdir(parents=True)
    shutil.copy(root / "opt/arc/baked-assets.json", vm2 / "opt/arc/baked-assets.json")
    process, remote, root2 = stage_run(tmp_path / "second", machine="vm2-unused", use_baked=True)
    process.kill()
    process.communicate(timeout=30)
    # the second staging made its own root; run the baked boot on vm2 with that spec
    spec = json.loads((root2 / "opt/arc/launch.json").read_text())
    shutil.copytree(root2 / "opt/arc/launch", vm2 / "opt/arc/launch")
    (vm2 / "opt/arc/launch.json").write_text(json.dumps(spec))
    env = {**os.environ, "ARC_NEBIUS_ROOT": str(vm2), "RCLONE_CONFIG_ARC_TYPE": "alias",
           "RCLONE_CONFIG_ARC_REMOTE": str(tmp_path / "second/s3")}
    process = subprocess.Popen([sys.executable, str(SCRIPTS / "nebius_runner.py"), str(vm2 / "opt/arc/launch.json")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    remote2 = tmp_path / "second/s3/arc-bucket/runs/validation/b2-e2e"
    try:
        assert wait_for(remote2 / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (remote2 / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    ready = json.loads((remote2 / "execution.json").read_text())["assets"]
    assert ready and all(a.get("baked") and a["fetch_seconds"] == 0.0 for a in ready), ready


@needs_rclone
def test_runner_refetches_when_the_image_record_differs(tmp_path):
    vm = tmp_path / "vm"
    (vm / "opt/arc").mkdir(parents=True)
    (vm / "opt/arc/baked-assets.json").write_text(json.dumps({"games": {"manifest_sha256": "old", "skip": []}}))
    process, remote, root = stage_run(tmp_path, use_baked=True)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (remote / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    ready = json.loads((remote / "execution.json").read_text())["assets"]
    assert not any(a.get("baked") for a in ready)


def test_stream_spec_and_streamer_calls(tmp_path, monkeypatch):
    assert nv.stream_spec({}) is None
    spec = nv.stream_spec({"nebius_stream": {"in": "opd-run/deltas", "in_seconds": 5, "out_seconds": 6}})
    assert spec == {"in": "opd-run/deltas", "in_seconds": 5.0, "out_seconds": 6.0}
    for bad in ({"in": "../x"}, {"in": "/abs"}, {"in": "a", "in_seconds": 1}):
        with pytest.raises(ValueError):
            nv.stream_spec({"nebius_stream": bad})
    monkeypatch.setattr(runner, "WORKING", tmp_path / "working")
    (tmp_path / "working/trials/t1/games/g1").mkdir(parents=True)
    calls = []
    fake = type("R", (), {})()
    fake.spec = {"stream": spec, "bucket": "b", "runs_prefix": "runs"}
    fake.remote_runs, fake.output = "arc:b/runs/validation/r1", tmp_path / "out"

    class Stop:
        n = 0

        def wait(self, seconds):
            Stop.n += 1
            return Stop.n > 3

    fake.stop = Stop()
    fake.rclone = lambda *args, **kw: calls.append(args) or type("P", (), {"returncode": 1, "stdout": ""})()
    fake.follow_source = lambda default, target: runner.Runner.follow_source(fake, default, target)  # no follow.json
    runner.Runner.streamer(fake)
    outs = [c for c in calls if len(c) > 2 and c[2] == "arc:b/runs/validation/r1/stream/trials"]
    ins = [c for c in calls if c[1] == "arc:b/runs/validation/opd-run/deltas"]
    assert outs and "*/games/*/rollout.jsonl" in outs[0] and "--local-no-check-updated" in outs[0]
    assert ins and ins[0][2] == str(tmp_path / "out/exchange-in/opd-run/deltas")


def test_streamer_pulls_live_knobs_on_change_only(tmp_path):
    spec = nv.stream_spec({"nebius_stream": {"in": "opd-run/deltas"}, "live_knobs": {"path": "x"}})
    assert spec["knobs"] is True
    fake = type("R", (), {})()
    fake.remote_runs, fake.output = "arc:b/runs/validation/r1", tmp_path / "out"
    body = {"v": '{"active_limit": 48}\n'}
    fake.rclone = lambda *args, **kw: type("P", (), {"returncode": 0 if body["v"] else 3, "stdout": body["v"]})()
    target = tmp_path / "out/exchange-in/knobs.json"
    runner.Runner.pull_knobs(fake)
    assert json.loads(target.read_text()) == {"active_limit": 48}
    body["v"] = "{bad"
    runner.Runner.pull_knobs(fake)  # unreadable: the local copy stays
    body["v"] = ""
    runner.Runner.pull_knobs(fake)  # missing: the local copy stays
    assert json.loads(target.read_text()) == {"active_limit": 48}
    body["v"] = '{"active_limit": 56, "kv_margin": 0.85}'
    runner.Runner.pull_knobs(fake)
    assert json.loads(target.read_text()) == {"active_limit": 56, "kv_margin": 0.85}


# ---- spec asset_overlap (MFU-2, 24 Oct): an asset fetched beside the bootstrap, usable only once sha256-verified ----
GATE_BOOTSTRAP = """
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("--bundle", "--working", "--output", "--games-dir", "--max-seconds"):
    p.add_argument(name)
p.add_argument("--trial", action="append")
a = p.parse_args()
assets = Path(os.environ["ARC_NEBIUS_ROOT"]) / "assets"
gate, target = assets / "flash-input/datasets/o/d.arc-gate.json", assets / "flash-input/datasets/o/d"
first = json.loads(gate.read_text())["state"]
early = []  # the target seen before the gate says verified
end = time.time() + 60
while time.time() < end:
    state = json.loads(gate.read_text())["state"]
    if target.exists() and state != "verified":
        early.append(state)
    if state != "fetching":
        break
    time.sleep(0.02)
data = (target / "weights.bin").read_bytes() if state == "verified" else None
Path(a.output).mkdir(parents=True, exist_ok=True)
Path(a.output, "bootstrap-results.json").write_text(json.dumps({"first": first, "final": state, "early": early,
                                                                "ok": data == b"w" * 4096}))
time.sleep(float(os.environ.get("GATE_TEST_LINGER", "0")))
sys.exit(0 if state == "verified" else 3)
"""


def slow_rclone(tmp_path, delay, hang=False):
    """An rclone on PATH that delays (or hangs without writing) only the copy into an .arc-partial folder."""
    real = shutil.which("rclone")
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    wrapper = folder / "rclone"
    then = "sleep 600" if hang else ":"
    wrapper.write_text(f"#!/bin/sh\ncase \"$*\" in *.arc-partial*) sleep {delay}; {then} ;; esac\nexec {real} \"$@\"\n")
    wrapper.chmod(0o755)
    return {"PATH": f"{folder}:{os.environ['PATH']}"}


def stage_overlap(tmp_path, monkeypatch, *, delay=2, hang=False, corrupt=False, **extra):
    for k, v in slow_rclone(tmp_path, delay, hang).items():
        monkeypatch.setenv(k, v)
    return stage_run(tmp_path, bootstrap=GATE_BOOTSTRAP, corrupt=corrupt, asset_overlap=["flash-input/datasets/o/d"],
                     **extra)


@needs_rclone
def test_overlapped_asset_is_readable_only_after_its_sha256(tmp_path, monkeypatch):
    process, remote, root = stage_overlap(tmp_path, monkeypatch, delay=3)
    output = process.communicate(timeout=120)[0]
    assert process.returncode == 0, output
    result = json.loads((root / "runs/validation/b2-e2e/arena/bootstrap-results.json").read_text())
    assert result == {"first": "fetching", "final": "verified", "early": [], "ok": True}  # started before the fetch
    gate = json.loads((root / "assets/flash-input/datasets/o/d.arc-gate.json").read_text())
    assert gate["state"] == "verified" and gate["mode"] == "sha256" and gate["overlapped"] is True
    assert not (root / "assets/flash-input/datasets/o/d.arc-partial").exists()
    ready = json.loads((remote / "execution.json").read_text())["assets"]
    assert any(a.get("overlapped") and a["prefix"] == "flash-input/datasets/o/d" for a in ready), ready
    lines = [json.loads(x) for x in output.splitlines() if x.startswith('{"event": "nebius_phase"')]
    assert [x["phase"] for x in lines][:3] == ["boot", "assets", "bootstrap"]


@needs_rclone
def test_overlapped_asset_with_a_bad_sha256_stops_the_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_TEST_LINGER", "0")
    process, remote, root = stage_overlap(tmp_path, monkeypatch, delay=2, corrupt=True)
    output = process.communicate(timeout=120)[0]
    assert process.returncode != 0, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and "overlapped asset flash-input/datasets/o/d" in final["error"], final
    assert "sha256 mismatch" in final["error"]
    gate = json.loads((root / "assets/flash-input/datasets/o/d.arc-gate.json").read_text())
    assert gate["state"] == "failed"
    assert not (root / "assets/flash-input/datasets/o/d").exists()  # the corrupt copy never became readable
    results = root / "runs/validation/b2-e2e/arena/bootstrap-results.json"
    if results.exists():  # the runner's kill may land before the job writes it
        assert json.loads(results.read_text())["early"] == []


@needs_rclone
def test_a_stalled_overlapped_fetch_fails_the_run(tmp_path, monkeypatch):
    process, remote, root = stage_overlap(tmp_path, monkeypatch, hang=True, asset_overlap_stall_seconds=2,
                                          asset_overlap_poll_seconds=0.3)
    output = process.communicate(timeout=120)[0]
    assert process.returncode != 0, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and "stalled" in final["error"], final
    assert json.loads((root / "assets/flash-input/datasets/o/d.arc-gate.json").read_text())["state"] == "failed"
    assert not (root / "assets/flash-input/datasets/o/d").exists()


@needs_rclone
def test_a_failure_while_the_job_still_runs_kills_it(tmp_path, monkeypatch):
    """The job here never waits on the gate (it lingers): the runner's own kill ends it, with the fetch's error."""
    monkeypatch.setenv("GATE_TEST_LINGER", "60")
    bootstrap = GATE_BOOTSTRAP.replace("    if state != \"fetching\":\n        break\n", "    break\n")
    for k, v in slow_rclone(tmp_path, 1).items():
        monkeypatch.setenv(k, v)
    began = time.time()
    process, remote, root = stage_run(tmp_path, bootstrap=bootstrap, corrupt=True,
                                      asset_overlap=["flash-input/datasets/o/d"])
    output = process.communicate(timeout=120)[0]
    assert process.returncode != 0 and time.time() - began < 50, output
    final = json.loads((remote / "nebius/final.json").read_text())
    assert "sha256 mismatch" in final["error"], final


def test_launcher_asset_overlap_is_opt_in_and_checked():
    assets = ["games", "flash-input/models/arc-heal/p75d4-sAD"]
    assert nv.asset_overlap_for({}, assets) == []
    assert nv.asset_overlap_for({"nebius_asset_overlap": [assets[1], assets[1]]}, assets) == [assets[1]]
    with pytest.raises(ValueError, match="does not fetch"):
        nv.asset_overlap_for({"nebius_asset_overlap": ["other"]}, assets)
    with pytest.raises(ValueError, match="list"):
        nv.asset_overlap_for({"nebius_asset_overlap": "games"}, assets)


# -- boot-disk pool (MFU-2, 24 Oct) -------------------------------------------------------------------------------
IMAGE = "computeimage-e00img"


class PoolCloud(FakeCloud):
    """FakeCloud with a disk table: list / get / update (labels-add) / create --async / delete."""

    def __init__(self, config, objects=None, disks=None):
        super().__init__(config, objects)
        self.disks = {d["metadata"]["id"]: d for d in (disks or [])}

    def run(self, argv, *, display=None, check=True, timeout=None):
        if "disk" not in argv[:4]:
            return super().run(argv, display=display, check=check, timeout=timeout)
        self.calls.append(argv)
        done = subprocess.CompletedProcess(argv, 0, "", "")
        verb = argv[argv.index("disk") + 1]

        def arg(flag):
            return argv[argv.index(flag) + 1]

        if verb == "list":
            done.stdout = json.dumps({"items": list(self.disks.values())})
        elif verb == "get":
            done.stdout = json.dumps(self.disks[arg("--id")])
        elif verb == "update":
            labels = self.disks[arg("--id")]["metadata"].setdefault("labels", {})
            labels.update(dict(kv.split("=", 1) for kv in arg("--labels-add").split(",")))
            done.stdout = json.dumps(self.disks[arg("--id")])
        elif verb == "create":
            new = f"computedisk-new{len(self.disks)}"
            self.disks[new] = {"metadata": {"id": new, "name": arg("--name"), "created_at": "z",
                                            "labels": dict(kv.split("=", 1) for kv in arg("--labels").split(","))},
                               "spec": {"type": arg("--type").upper(), "size_gibibytes": arg("--size-gibibytes")},
                               "status": {"state": "CREATING"}}
        elif verb == "delete":
            self.disks.pop(arg("--id"))
        return done


def pool_disk(i, state="READY", pool="free", gib=1116, attached=None, claimed=0, created="a"):
    labels = {**nv.pool_labels(IMAGE, gib, "network_ssd_non_replicated"), nv.POOL_STATE: pool}
    if claimed:
        labels[nv.POOL_CLAIMED] = str(claimed)
    return {"metadata": {"id": f"computedisk-p{i}", "name": f"p{i}", "labels": labels, "created_at": created},
            "spec": {"type": "NETWORK_SSD_NON_REPLICATED", "size_gibibytes": str(gib)},
            "status": {"state": state, **({"read_write_attachment": attached} if attached else {})}}


def test_pool_fill_keeps_room_for_one_more_boot_disk_under_the_disk_quota():
    config = {**nv.check_config(CONFIG), "disk_quota_gib": 5120}
    other = {"metadata": {"id": "computedisk-vm", "labels": {}},
             "spec": {"type": "NETWORK_SSD_NON_REPLICATED", "size_gibibytes": "1800"}, "status": {"state": "READY"}}
    cloud = PoolCloud(config, disks=[other, pool_disk(1, pool="claimed", attached="computeinstance-x")])
    out = cloud.pool_fill(IMAGE, 1116, "network_ssd_non_replicated", 2, now=5)
    assert len(out["creating"]) == 0 and "disk quota" in out["skipped"]  # 2916 + 2x1116 > 5120
    cloud.disks["computedisk-vm"]["spec"]["size_gibibytes"] = "1000"
    out = cloud.pool_fill(IMAGE, 1116, "network_ssd_non_replicated", 2, now=6)
    assert len(out["creating"]) == 1 and "disk quota" in out["skipped"]  # 2116 -> create -> 3232 + 2232 > 5120

    config = nv.check_config(CONFIG)
    argv = nv.create_command(config, "n", {}, "UD", image_id=IMAGE, existing_disk="computedisk-p1")
    assert argv[argv.index("--boot-disk-existing-disk-id") + 1] == "computedisk-p1"
    assert argv[argv.index("--boot-disk-attach-mode") + 1] == "READ_WRITE"
    assert not [a for a in argv if a.startswith("--boot-disk-managed-disk")]


def test_pool_size_needs_a_baked_image():
    assert nv.pool_size(STUDY) == 0
    assert nv.pool_size({**STUDY, "nebius_image_id": IMAGE, "nebius_boot_disk_pool": 1}) == 1
    with pytest.raises(ValueError, match="needs nebius_image_id"):
        nv.pool_size({**STUDY, "nebius_boot_disk_pool": 1})
    with pytest.raises(ValueError, match="0-4"):
        nv.pool_size({**STUDY, "nebius_image_id": IMAGE, "nebius_boot_disk_pool": 9})


def test_pool_take_claims_the_oldest_free_disk_of_the_same_kind():
    config = nv.check_config(CONFIG)
    disks = [pool_disk(1, created="b"), pool_disk(2, created="a"), pool_disk(3, gib=558, created="0"),
             pool_disk(4, state="CREATING", created="0"), pool_disk(5, attached="computeinstance-x", created="0"),
             pool_disk(6, pool="claimed", created="0")]
    cloud = PoolCloud(config, disks=disks)
    got = cloud.pool_take(IMAGE, 1116, "network_ssd_non_replicated", "b2-run", now=1000)
    assert got == "computedisk-p2"
    labels = cloud.disks["computedisk-p2"]["metadata"]["labels"]
    assert labels[nv.POOL_STATE] == "claimed" and labels[nv.POOL_RUN] == "b2-run" and labels[nv.POOL_CLAIMED] == "1000"
    assert cloud.pool_take(IMAGE, 1116, "network_ssd_non_replicated", "b2-two", now=1001) == "computedisk-p1"
    assert cloud.pool_take(IMAGE, 1116, "network_ssd_non_replicated", "b2-three", now=1002) is None


def test_pool_fill_counts_creating_disks_and_creates_async():
    config = nv.check_config(CONFIG)
    cloud = PoolCloud(config, disks=[pool_disk(1, state="CREATING"), pool_disk(2, pool="claimed")])
    out = cloud.pool_fill(IMAGE, 1116, "network_ssd_non_replicated", 2, now=5)
    assert out["had"] == 1 and len(out["creating"]) == 1
    create = next(c for c in cloud.calls if "create" in c)
    assert "--async" in create and create[create.index("--source-image-id") + 1] == IMAGE
    assert create[create.index("--size-gibibytes") + 1] == "1116"
    made = [d for d in cloud.disks.values() if d["metadata"]["id"].startswith("computedisk-new")]
    assert made[0]["metadata"]["labels"] == {**nv.pool_labels(IMAGE, 1116, "network_ssd_non_replicated"),
                                             nv.POOL_STATE: "free"}
    assert cloud.pool_fill(IMAGE, 1116, "network_ssd_non_replicated", 2, now=6)["creating"] == []


def test_pool_gc_deletes_only_detached_claimed_disks_past_the_grace():
    config = nv.check_config(CONFIG)
    now = 100_000
    disks = [pool_disk(1, pool="claimed", claimed=now - 5000),  # its VM is gone: delete
             pool_disk(2, pool="claimed", claimed=now - 10),  # a create may still be attaching it
             pool_disk(3, pool="claimed", claimed=now - 5000, attached="computeinstance-x"),  # in use
             pool_disk(4, claimed=0)]  # free
    cloud = PoolCloud(config, disks=disks)
    assert cloud.pool_gc(now) == ["computedisk-p1"]
    assert set(cloud.disks) == {"computedisk-p2", "computedisk-p3", "computedisk-p4"}


def test_launch_boots_from_a_pool_disk_and_refills_after_the_create(tmp_path):
    make_ledger(tmp_path, cap=2000)
    study = {**STUDY, "notebook_seconds": 3000, "wall_seconds": 2400, "nebius_image_id": IMAGE,
             "nebius_boot_disk_pool": 1, "nebius_boot_disk_gib": 1116,
             "b200": {**STUDY["b200"], "stop": {**STUDY["b200"]["stop"], "wall_seconds": 2400}}}
    bundle = make_bundle(tmp_path, study)
    config = nv.check_config(CONFIG)
    objects = {**manifests(nv.required_assets(study)), "runs/validation/b2-p1/nebius/status.json": {"phase": "boot"}}
    cloud = PoolCloud(config, objects, disks=[pool_disk(1)])
    record = nv.launch(tmp_path, config, cloud, run_id="b2-p1", kind="arena", test="b200-run", bundle=bundle,
                       now=1_000_000.0, wait=False)
    create = next(c for c in cloud.calls if c[1:4] == ["compute", "instance", "create"] or
                  c[1:4] == ["--profile", "arc", "compute"] or ("instance" in c and "create" in c))
    assert create[create.index("--boot-disk-existing-disk-id") + 1] == "computedisk-p1"
    assert record["attempts"][-1]["pool_disk"] == "computedisk-p1"
    assert record["spec"]["boot_disk_pool"] == 1 and len(record["pool_fill"]["creating"]) == 1
    order = [i for i, c in enumerate(cloud.calls) if "disk" in c and "create" in c]
    assert order and order[0] > cloud.calls.index(create)  # the refill starts after the VM create
    # An empty pool: the managed disk from the image, as before, and the refill still runs.
    cloud2 = PoolCloud(config, dict(objects), disks=[])
    record2 = nv.launch(tmp_path, config, cloud2, run_id="b2-p2", kind="arena", test="b200-run", bundle=bundle,
                        now=1_000_100.0, wait=False)
    create2 = next(c for c in cloud2.calls if "instance" in c and "create" in c)
    assert "--boot-disk-existing-disk-id" not in create2
    assert create2[create2.index("--boot-disk-managed-disk-source-image-id") + 1] == IMAGE
    assert record2["attempts"][-1]["pool_disk"] is None and len(record2["pool_fill"]["creating"]) == 1
def test_allowance_gate_checks_the_launching_arm(tmp_path, monkeypatch):
    make_ledger(tmp_path, cap=100)
    (tmp_path / "docs/html").mkdir(parents=True)
    (tmp_path / "docs/html/arm-state.json").write_text(json.dumps({"run_arms": {"b2-": "arc3sub", "arc2-": "arc2"}}))
    (tmp_path / "docs/modal-queue.json").write_text(json.dumps({"live": [
        {"run": "b2-live-1", "arm": "arc3sub", "platform": "nebius", "gpu": "RTX PRO 6000"}]}))
    from arc_infra.runops import records

    records.migrate(tmp_path, records.arm_rules(tmp_path))
    budget = json.loads((tmp_path / "docs/budget.json").read_text())
    budget.update(allowances={"arc3sub": {"gpus": {"RTX PRO 6000": 1}}, "arc2": {"gpus": {"RTX PRO 6000": 1}}},
                  caps={"gpus": {"RTX PRO 6000": 8}, "usd": {"nebius": 100}})
    (tmp_path / "docs/budget.json").write_text(json.dumps(budget))
    entry = {"run_id": "b2-n9", "state": "reserved", "gpu": "RTX-PRO-6000", "charged_bound_usd": 3.0,
             "rate_usd_per_hour": 1.254}
    with pytest.raises(records.GateError, match="arc3sub: 1 RTX PRO 6000 live"):
        nv.reserve(tmp_path, dict(entry))
    monkeypatch.setenv("ARC_ALLOWANCE_EXEMPT", "c100 300K quality read")
    assert nv.reserve(tmp_path, dict(entry)) == 3.0
    monkeypatch.delenv("ARC_ALLOWANCE_EXEMPT")
    assert records.get("b2-n9", tmp_path)["queue"]["row"]["allowance_exempt"].startswith("c100")
    with pytest.raises(records.GateError, match="arc3sub: 1 RTX PRO 6000 live"):  # the exempt read is not counted
        nv.reserve(tmp_path, {**entry, "run_id": "b2-n10"})
    assert nv.reserve(tmp_path, {**entry, "run_id": "arc2-x1"}) == 3.0  # another arm's use is not its business
    with pytest.raises(records.GateError, match="arc2: 1 RTX PRO 6000 live"):  # counted at once, before reconcile
        nv.reserve(tmp_path, {**entry, "run_id": "arc2-x2"})
    assert [r["run_id"] for r in json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"]] == \
        ["b2-n9", "arc2-x1"]
    assert records.get("arc2-x1", tmp_path)["ledger"][-1]["block"] == "nebius"  # written to the record, not in place
    settled = nv.settle_entry(tmp_path, "arc2-x1", "complete", 600)
    assert settled["state"] == "complete" and settled["charged_bound_usd"] < 3.0
    view = json.loads((tmp_path / "docs/spend.json").read_text())["nebius"]["runs"]
    assert next(r for r in view if r["run_id"] == "arc2-x1")["state"] == "complete"
    with pytest.raises(StopIteration):
        nv.settle_entry(tmp_path, "zz-none", "complete", 1)
    with pytest.raises(RuntimeError, match="No arm"):
        nv.reserve(tmp_path, {**entry, "run_id": "zz-1"})


# ---- spec asset_late: the read VM boots while the checkpoint export runs -------------------------------
LATE = "flash-input/datasets/o/d"


def stage_late(tmp_path, *, rule=None, after=None, signal_body=None):
    """stage_run with the overlay's manifest held back: written ``after`` s later (None = never), or a failed signal."""
    process, remote, root = None, None, None
    held = tmp_path / "held.json"
    bucket = tmp_path / "s3/arc-bucket"

    def release():
        manifest = bucket / "assets-manifests" / f"{nv.asset_slug(LATE)}.json"
        if after is not None:
            time.sleep(after)
            os.replace(held, manifest)
        if signal_body is not None:
            time.sleep(1)
            (bucket / "assets-signals").mkdir(exist_ok=True)
            (bucket / "assets-signals/d.failed.json").write_text(signal_body)

    real = subprocess.Popen

    def popen(argv, **k):  # stage_run writes the manifests, then starts the runner: hold the late one back first
        manifest = bucket / "assets-manifests" / f"{nv.asset_slug(LATE)}.json"
        os.replace(manifest, held)
        threading.Thread(target=release, daemon=True).start()
        return real(argv, **k)

    subprocess.Popen = popen
    try:
        process, remote, root = stage_run(tmp_path, asset_late={LATE: rule or {"wait_seconds": 60, "poll_seconds": 0.3}})
    finally:
        subprocess.Popen = real
    return process, remote, root


@needs_rclone
def test_a_late_asset_is_waited_for_then_fetched_and_checked_before_the_bootstrap(tmp_path):
    process, remote, root = stage_late(tmp_path, after=3)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (remote / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    events = [json.loads(x) for x in output.splitlines() if x.startswith("{")]
    order = [(e["event"], e.get("phase") or e.get("prefix")) for e in events
             if e["event"] in ("nebius_phase", "asset_ready", "asset_late_ready")]
    assert order.index(("asset_ready", "games")) < order.index(("nebius_phase", "asset_wait"))
    assert order.index(("asset_late_ready", LATE)) < order.index(("asset_ready", LATE))
    assert order.index(("asset_ready", LATE)) < order.index(("nebius_phase", "bootstrap"))
    execution = json.loads((remote / "execution.json").read_text())
    assert execution["asset_waits"][LATE] >= 2 and (root / "assets" / LATE / "weights.bin").stat().st_size == 4096
    assert any(a["prefix"] == LATE and a["mode"] == "sha256" for a in execution["assets"])


@needs_rclone
def test_a_late_asset_past_its_ceiling_or_with_a_failed_export_ends_the_run(tmp_path):
    began = time.time()
    process, remote, root = stage_late(tmp_path / "a", rule={"wait_seconds": 3, "poll_seconds": 0.3})
    output = process.communicate(timeout=90)[0]
    final = json.loads((remote / "nebius/final.json").read_text())
    assert process.returncode != 0 and final["state"] == "failed", output
    assert "no manifest within 3 s" in final["error"] and time.time() - began < 60
    assert not (root / "assets" / LATE).exists() and not (remote / "arena/live/progress.json").exists()

    process, remote, root = stage_late(tmp_path / "b", signal_body='{"step": 1293, "error": "oom"}',
                                       rule={"wait_seconds": 600, "poll_seconds": 0.3})
    output = process.communicate(timeout=90)[0]
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and "the export failed" in final["error"] and "oom" in final["error"], output


def test_launcher_asset_late_rules_disk_and_reservation(tmp_path):
    assets = ["games", "flash-input/models/arc-heal/p75d4-sAD"]
    late = {assets[1]: {"wait_seconds": 3600, "gib": 33}}
    assert nv.asset_late_for({}, assets) == {}
    assert nv.asset_late_for({"nebius_asset_late": late}, assets) == late
    for bad, match in (({"other": {"wait_seconds": 600, "gib": 1}}, "does not fetch"),
                       ({assets[1]: {"wait_seconds": 30, "gib": 1}}, "wait_seconds"),
                       ({assets[1]: {"wait_seconds": 600}}, "gib"), ([assets[1]], "map")):
        with pytest.raises(ValueError, match=match):
            nv.asset_late_for({"nebius_asset_late": bad}, assets)
    with pytest.raises(ValueError, match="also in nebius_asset_overlap"):
        nv.asset_late_for({"nebius_asset_late": late, "nebius_asset_overlap": [assets[1]]}, assets)

    class Cloud:
        dry_run = False

        def read_json(self, prefix, name):
            assert "p75d4" not in name  # not in the bucket yet: its declared size counts
            return {"files": {"f": {"size": 166 * 2 ** 30}}}

    config = {"boot_disk_gib": 372, "boot_disk_type": "network_ssd_non_replicated"}
    assert nv.boot_disk_for(config, Cloud(), assets, late={assets[1]: {"wait_seconds": 600, "gib": 100}}) == 372

    # A launch whose overlay is not mirrored yet: allowed when declared late, the wait reserved on top.
    make_ledger(tmp_path, cap=2000)
    config = nv.check_config(CONFIG)
    objects = {**manifests([p for p in nv.required_assets(STUDY) if not p.endswith(OVERLAY)]),
               "runs/validation/b2-n1/nebius/status.json": {"phase": "bootstrap"}}
    overlay = f"flash-input/datasets/{OVERLAY}"
    with pytest.raises(RuntimeError, match="not mirrored"):
        nv.launch(tmp_path, config, FakeCloud(config, objects), run_id="b2-n0", kind="arena", test="b200-run",
                  now=1000.0, approval=APPROVED, bundle=make_bundle(tmp_path))
    study = {**STUDY, "nebius_asset_late": {overlay: {"wait_seconds": 3600, "gib": 33}}}
    cloud = FakeCloud(config, objects)
    record = nv.launch(tmp_path, config, cloud, run_id="b2-n1", kind="arena", test="b200-run", now=1000.0,
                       approval=APPROVED, bundle=make_bundle(tmp_path, study=study))
    assert spec_of(cloud)["asset_late"] == {overlay: {"wait_seconds": 3600, "gib": 33}}
    assert record["max_seconds"] == 86400 + 2400 + 300 + 3600
    assert record["deadline_epoch"] == 1000.0 + 86400 + 2400 + 300 + 3600


@needs_rclone
def test_a_failed_signal_after_the_late_asset_landed_stops_the_running_job(tmp_path):
    """SFT's byte checks run after the manifest: their failed signal stops a read already serving."""
    process, remote, root = stage_late(tmp_path, after=1)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (tmp_path / "s3/arc-bucket/assets-signals").mkdir(exist_ok=True)
        (tmp_path / "s3/arc-bucket/assets-signals/d.failed.json").write_text('{"error": "dn bad 3"}')
        began = time.time()
        output = process.communicate(timeout=60)[0]
    finally:
        process.kill()
    final = json.loads((remote / "nebius/final.json").read_text())
    assert final["state"] == "failed" and "the export failed" in final["error"] and "dn bad 3" in final["error"], output
    assert time.time() - began < 30


@needs_rclone
def test_a_late_asset_is_fetched_beside_a_slow_base_fetch(tmp_path, monkeypatch):
    """the 1293 overlay was ready 7 min before the base fetch ended; it now goes at once."""
    real = shutil.which("rclone")
    (tmp_path / "bin").mkdir()
    wrapper = tmp_path / "bin/rclone"
    wrapper.write_text(f"#!/bin/sh\ncase \"$*\" in copy\\ *assets/games*) sleep 4 ;; esac\nexec {real} \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    process, remote, root = stage_late(tmp_path, after=0.5)
    try:
        assert wait_for(remote / "arena/live/progress.json"), process.communicate(timeout=5)[0]
        (remote / "arena/live/stop-request").write_text(json.dumps({"reason": "test"}))
        output = process.communicate(timeout=90)[0]
    finally:
        process.kill()
    assert process.returncode == 0, output
    ready = [json.loads(x)["prefix"] for x in output.splitlines() if x.startswith('{"event": "asset_ready"')]
    assert ready == [LATE, "games"], output  # the overlay landed and was checked while games was still copying
