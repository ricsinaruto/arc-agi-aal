"""Nebius AI Cloud backend for the arena validation bundles: one RTX PRO 6000 VM per run, a fixed (on-demand)
instance unless --preemptible is given ("fixed instance for everything"; B200 stays on Modal).

scripts/modal_validation.py is untouched and stays the default backend. Same interface (environment + --run-id):

    ARC_VALIDATION_ROOT=. ARC_VALIDATION_KIND=arena ARC_VALIDATION_TEST=b200-run \
    ARC_VALIDATION_BUNDLE=dist/<bundle> python scripts/nebius_validation.py --run-id <id> [--dry-run]

Operator notes
--------------
First use: `setup` (account commands, config template) -> `set-cap` -> `preflight` -> the `mirror-plan` Modal job
-> a b200-smoke bundle -> long runs. Cloud-validated 7 October 2026: preflight, mirror, smoke v3 (boot 175 s, asset
fetch 280 s, self-delete). Relaunch, resume and the controller: tested offline only, not yet against the cloud.
What a launch does: checks the bundle and the asset manifests, reserves the bound in docs/spend.json ("nebius"
section, own cap), uploads bundle + runner + spec to s3://<bucket>/launch/<run_id>/, creates the VM with a cloud-init
script, waits for its first heartbeat (deletes the VM if none arrives) and writes docs/validation-nebius/
nebius-<run_id>.json. On the VM (scripts/nebius_runner.py): assets from the bucket to /assets (manifest-checked),
/kaggle/input -> /assets/flash-input, bundle at /opt/arc-validation, the bundle's bootstrap with Modal's arguments
and environment in a loopback-only network namespace, output in /runs/validation/<run_id> copied to
s3://<bucket>/runs/validation/<run_id>/ every 60 s (same layout as the arc-prolong-runs volume).

    status RUN_ID            instance state, phase, live progress, attempts, spend so far; settles a finished run
    pull RUN_ID [--live]     copy the run's bucket folder to runs/nebius/validation/<run_id> (--live: arena/live only)
    stop RUN_ID --reason ..  graceful stop: validation/<run>/arena/live/stop-request, seen within 30 s
    watch [RUN_ID] [--no-relaunch] poll (every active run without an id); relaunch after a preemption; settle
    controller up|status|down      the always-on watcher: a CPU VM running `watch --controller` over every run
    list / reap [--yes]      VMs labelled arc-run; reap deletes stopped, overdue and finished ones (dry without --yes)
    relaunch RUN_ID [--fresh]      the pending relaunch now, by hand
    preflight --run-id ID CPU VM: service account, SecretStash, bucket, isolation, self-delete
    set-cap USD --authorization "<the user's words>"   the Nebius cap (launches refuse to run without it)
    mirror-plan              the bundle's asset prefixes, the ones not in the bucket and the mirror command for them
    setup                    the one-time account commands, from the config

Fixed instance (the default): list + boot disk, one attempt, the bound is that attempt (timeout + setup
allowance + finalize); no relaunch policy, no controller, no checkpoints unless --checkpoint-seconds is given (they
are written to the bucket; nothing resumes from them by itself). A create that finds no capacity or no quota fails
cleanly: nothing left, nothing charged. Several runs in parallel: one VM each; a launch refuses when the project's
RTX PRO 6000 quota (read from `quotas quota-allowance list`) or config max_parallel_gpu_vms has no room, or when
no public IPv4 address is free (vpc.ipv4-address.public.count: every VM takes one; limit 3 on 7 October).
Config public_ip false creates VMs with a private address only: everything they reach (apt, PyPI, the CLI download,
SecretStash, the bucket, the compute API for the self-delete) is outbound and goes through the network's default
egress gateway (docs.nebius.com/vpc/routing/overview); no ssh then. Not yet run against the cloud: `preflight` with
the switch set is the test (cents, CPU VM, same paths).
--preemptible: the spot price and everything under "Preemption" below.

Cost guards, in order: the VM's run unit ends -> ExecStopPost deletes the VM (managed disk included); a systemd timer
deletes it at the deadline (never past the run's reservation); a launch without a heartbeat is deleted by the
launcher; `reap` is the outside net (a VM whose cloud-init never ran is only caught by the last two). If the delete
call fails the VM powers off: no compute charge, but the disk bills until `reap`.
Preemption (--preemptible runs): Nebius stops the VM after SIGTERM + 60 s. The runner asks the trial for a last
checkpoint, uploads it and writes nebius/final.json "interrupted"; the watcher (the controller VM, or `watch` here)
deletes the VM and starts the next attempt: a bundle with src/arena/checkpoint.py resumes in the same output folder
from its newest checkpoint (attempt kind "resumed"; arena.json `checkpoint` says from which); any other bundle, or a
run cut before its first checkpoint was uploaded, restarts from zero under <run>-a<N> (kind "fresh"; never merged
with the partial folder). No relaunch after any other failure, past max_attempts (4), or beyond the reservation.
The ledger is local, so a launch reserves the whole run: one full attempt + (max_attempts - 1) x (relaunch setup
900 s + lost work: checkpoint interval + 120 s, or the whole run for a bundle that cannot resume); every VM's deadline
stays inside what is left of it. Play time (the launched trials' wall, a competition bundle's preview) above 2 h
needs --reason "<why the mechanism needs it>", above 4 h --approval "<the user's words>"; both go into the receipt.
Without spot capacity a create fails after Nebius's scheduling wait (~6 min, exit 65 "VM schedule timeout ...
insufficient hardware resources"; seen 7 October on a first create). On any attempt, the first included, the
unscheduled instance and its boot disk are deleted, nothing is charged (a stopped VM bills its disk only) and the run
stays relaunch_pending: the controller or `watch` retries for capacity_retry_seconds (config, default 6 h), then the
run ends as relaunch_failed: no on-demand fallback (a run over 4 h: ask the user).
Where the watcher lives: `controller up` (cpu-d3 2 vCPU / 8 GiB, reserved for --hours, deletes itself after
an idle hour) works from launch/<run>/record.json with the service account, so relaunches do not depend on this
laptop; `watch` here does the same while it runs and stands back while a controller heartbeat is fresh (15 min: one
create can block that side for Nebius's whole scheduling wait). Keep `watch` (no run id) running as the fallback: it
takes over when the controller reports controller_unfit (its CLI lacks a create flag, or 3 identical create errors)
or a run has 3 non-capacity create failures. Every VM installs this side's CLI version (config cli_version, read
from the local CLI at launch / controller up): the image's own was older than the laptop's on 7 October.
Config: ~/.nebius/arc.json (ids only, no secrets; `setup` prints the template). The bucket key lives in SecretStash
(VM side) and in the local rclone remote (operator side); this script never reads it.
Not supported here: sglang / vllm kinds, teacher runs (network), tests that write to the assets volume, B200.
"""

import argparse
import ast
import base64
import contextlib
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
B200_TESTS = ("b200-smoke", "b200-run")
# docs.nebius.com/compute/resources/pricing, checked 7 October 2026 (USD; the listed GPU hour has no separate
# vCPU / RAM items for this platform). Spot prices move every 15 minutes inside a range that tops out below the
# on-demand price (docs.nebius.com/signup-billing/pricing-policy), so a bound uses the policy's max price or,
# when following the spot price, the on-demand price.
PRICES = {"checked": "2026-10-07", "url": "https://docs.nebius.com/compute/resources/pricing",
          "gpu_hour": {"on_demand": 1.80, "preemptible_list": 0.95},
          "disk_gib_month": {"network_ssd": 0.071, "network_ssd_non_replicated": 0.053, "network_ssd_io_m3": 0.118},
          "cpu_d3": {"vcpu_hour": 0.015, "gib_hour": 0.0045}}
PLATFORMS = {"us-central1": "gpu-rtx6000", "uk-south2": "gpu-rtx6000-a", "eu-south1": "gpu-rtx6000-a"}
DEFAULTS = {"profile": None, "preset": "1gpu-24vcpu-218gb", "image_family": "ubuntu24.04-cuda13.0",
            "boot_disk_type": "network_ssd_non_replicated", "boot_disk_gib": 744, "rclone_remote": "nebius-arc",
            "pricing_policy_id": None, "spot_max_price_usd_per_gpu_hour": None, "follow_spot_price": False,
            "setup_allowance_seconds": 2400, "verify": "sha256", "ssh_user": None, "ssh_public_key": None,
            "heartbeat_seconds": 720,
            # Preemption policy: attempts per run (the first included), checkpoint interval of a resumable bundle,
            # boot + asset fetch + server start of a relaunch (measured 7 October: 175 + 280 s before the bootstrap),
            # and how long a relaunch waits for spot capacity.
            "max_attempts": 4, "checkpoint_seconds": 600, "relaunch_setup_seconds": 900,
            "capacity_retry_seconds": 6 * 3600,
            # Our own ceiling on GPU VMs at once (None: the project quota alone decides).
            "max_parallel_gpu_vms": None,
            # False: a private address only (outbound through the network's default egress gateway; no ssh).
            "public_ip": True,
            # The create call waits for the VM to be scheduled (Nebius gives up after ~6 min); a hung CLI ends here.
            "create_timeout_seconds": 900}
REQUIRED = ("project_id", "region", "subnet_id", "service_account_id", "bucket", "s3_secret_id")
FINALIZE_SECONDS = 300
BASE_ASSETS = ("flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4",
               "flash-input/datasets/keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
               "flash-input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1", "games")
BUNDLE_IGNORE = ("dataset-upload", "source.zip")  # and *.ipynb: modal_validation's add_local_dir ignore list
# The Modal image's apt set plus what debian_slim's Python brings (headers, venv) and this backend's tools.
PACKAGES = ("python3.12-dev", "python3.12-venv", "build-essential", "git", "binutils", "libseccomp2",
            "libglib2.0-0t64", "libgl1", "libnuma1", "libgomp1", "rclone", "curl", "iproute2", "util-linux", "dmidecode")
RUNS_PREFIX, ASSETS_PREFIX, MANIFESTS_PREFIX = "runs", "assets", "assets-manifests"
RECORDS = "docs/validation-nebius"
CANCEL_MARKER = "nebius/cancelled.json"  # in the attempt's output folder; the runner refuses to start under it
BAKE_KEEP_SECONDS = 6 * 3600
TERMINAL = ("baked", "complete", "failed", "timeout", "preempted", "reaped", "lost", "setup_failed", "preflight_ok",
            "relaunch_failed", "stopped")
# on play time: above 2 h a recorded reason, above 4 h the user's approval before launch.
LONG_RUN_SECONDS, APPROVAL_SECONDS = 7200, 14400
NO_CAPACITY = re.compile(r"schedule timeout|insufficient\s+(\w+\s+)?resources|not enough resources|"
                         r"no (available )?capacity|out of capacity", re.I)
NO_QUOTA = re.compile(r"quota", re.I)
ADDRESSES = "vpc.ipv4-address.public.count"
NOISE = re.compile(r"(?i)\b(request|trace|operation)[ _-]?id\b\s*[:=]?\s*\S*")
SIGNAL = re.compile(r"(?i)error|fail|violation|quota|limit|denied|unknown|invalid|not found|timeout|insufficient|"
                    r"exhaust|capacity|resources|exists")


def explain(stderr, limit=600):
    """What a failed CLI call said, without its request and trace ids: the lines that name the failure, else the
    tail. Stable across retries, so two identical failures compare equal."""
    lines = [NOISE.sub("", line).strip(" ,;|") for line in (stderr or "").splitlines()]
    lines = [line for line in lines if line]
    signal = [line for line in lines if SIGNAL.search(line)]
    return " | ".join(signal)[:limit] if signal else " | ".join(lines)[-limit:]


def said(exc, limit=400):
    """The provider's own words in a failed call's exception (after ``exited N:``)."""
    text = str(exc)
    found = re.search(r"exited -?\d+: (.*)", text, re.S)
    return (found.group(1) if found else text)[:limit].strip()
CLAIM_SECONDS = 900  # one side's create in flight: the other (controller / laptop) keeps off the same attempt
SELF_SIZED = ("level-bench", "expert-usage", "prune-quality", "expert-quant")  # trials that play what the timeout leaves
CONTROLLER = "launch/controller"  # the controller VM's heartbeat lives here

BOOTSTRAP = r"""#!/bin/bash
# Rendered by scripts/nebius_validation.py for one run. Holds no secrets.
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/root/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
mkdir -p /opt/arc/launch /var/log/arc /run/arc
exec >>/var/log/arc/bootstrap.log 2>&1
echo "arc bootstrap $(date -u +%FT%TZ)"
RUN_ID=@RUN_ID@
BUCKET=@BUCKET@
ENDPOINT=@ENDPOINT@
REGION=@REGION@
SECRET_ID=@SECRET_ID@
PROJECT_ID=@PROJECT_ID@
LAUNCH_PREFIX=@LAUNCH_PREFIX@
RUNS_PREFIX=@RUNS_PREFIX@
DEADLINE_EPOCH=@DEADLINE_EPOCH@
RUNNER_SHA256=@RUNNER_SHA256@
PACKAGES=@PACKAGES@

# 0. No automatic package upgrades or service restarts during a run (12 Oct, b2d-v82srv-eq v3: arc-run.service was
#    stopped and started again 35 min after boot with no reboot or preemption, the signature of needrestart after an
#    unattended upgrade; the unit's stop deletes the VM).
systemctl stop apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service \
  unattended-upgrades.service 2>/dev/null
systemctl mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service 2>/dev/null
dpkg --configure -a 2>/dev/null
mkdir -p /etc/needrestart/conf.d
cat >/etc/needrestart/conf.d/90-arc.conf <<'EOF'
$nrconf{restart} = 'l';
$nrconf{override_rc}{qr(^arc-.*\.service$)} = 0;
EOF

# 1. Cost guard first: the VM deletes itself at the deadline whatever happens below.
umask 077
cat >/run/arc/nebius.yaml <<EOF
default: arc
profiles:
    arc:
        endpoint: api.nebius.cloud
        token-file: /run/arc/iam-token
        parent-id: $PROJECT_ID
EOF
umask 022
cat >/usr/local/bin/arc-nebius <<'EOF'
#!/bin/bash
# The Nebius CLI as the VM's service account: a fresh metadata-service token for every call (the CLI needs a
# profile; a token-file profile reads the file each time).
export HOME=/root PATH=/root/.nebius/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
umask 077
curl -sf -m 10 -H 'Metadata: true' http://metadata.nebius.internal/v1/iam/sa/token/access_token \
  >/run/arc/iam-token.new && [ -s /run/arc/iam-token.new ] && mv /run/arc/iam-token.new /run/arc/iam-token
[ -s /run/arc/iam-token ] || cp /mnt/cloud-metadata/token /run/arc/iam-token 2>/dev/null
BIN=/opt/arc/bin/nebius; [ -x "$BIN" ] || BIN=nebius  # the launcher's CLI version when the bootstrap installed it
exec "$BIN" -c /run/arc/nebius.yaml --no-browser --no-check-update "$@"
EOF
cat >/usr/local/bin/arc-self-delete <<'EOF'
#!/bin/bash
# Delete this VM and its managed disk; power off (no compute charge) if the API refuses.
# A bake (study nebius_bake) keeps the disk for its image: power off only. The marker is on tmpfs (/run), so a VM
# booted from the image never inherits it. Its logs are cleared (already uploaded) so the image does not carry them.
[ -f /run/arc/bake-keep ] && { rm -f /var/log/arc/*.log; : > /var/log/cloud-init-output.log; sync; shutdown -h now; exit 0; }
for attempt in 1 2 3 4 5 6; do
  ID=$(curl -sf -m 10 -H 'Metadata: true' http://metadata.nebius.internal/v1/instance-data/id)
  [ -n "$ID" ] && /usr/local/bin/arc-nebius compute instance delete --id "$ID" --async && exit 0
  sleep 15
done
shutdown -h now
EOF
chmod 0755 /usr/local/bin/arc-nebius /usr/local/bin/arc-self-delete
LEFT=$(( DEADLINE_EPOCH - $(date +%s) ))
[ "$LEFT" -lt 60 ] && LEFT=60
systemd-run --unit arc-deadline --on-active="${LEFT}s" /usr/local/bin/arc-self-delete \
  || { (sleep "$LEFT"; /usr/local/bin/arc-self-delete) & }

fail() {
  echo "arc bootstrap failed: "
  if command -v rclone >/dev/null && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    printf '{"state": "setup_failed", "run_id": "%s", "error": "%s", "ended_epoch": %s}\n' \
      "$RUN_ID" "" "$(date +%s)" >/run/arc/final.json
    BASE="arc:$BUCKET/$RUNS_PREFIX/validation/$RUN_ID/nebius"
    rclone copyto /var/log/arc/bootstrap.log "$BASE/bootstrap.log" --retries 3
    rclone copyto /run/arc/final.json "$BASE/final.json" --retries 3
  fi
  /usr/local/bin/arc-self-delete
  exit 1
}

# 2. Nebius CLI. The launcher's own version when it names one: the image's can be older and lack flags (7 October,
#    controller VM: "unknown flag: --spot-pricing-policy-id"). Else the preinstalled one
#    (docs.nebius.com/cli/compute-vm) or the official installer.
CLI_VERSION=@CLI_VERSION@
if [ -n "$CLI_VERSION" ]; then
  mkdir -p /opt/arc/bin
  for attempt in 1 2 3; do
    curl -fsSL -m 120 "https://storage.eu-north1.nebius.cloud/cli/release/$CLI_VERSION/linux/$(uname -m)/nebius" \
      -o /opt/arc/bin/nebius.new && chmod 0755 /opt/arc/bin/nebius.new \
      && [ "$(/opt/arc/bin/nebius.new version 2>/dev/null | head -1)" = "$CLI_VERSION" ] \
      && mv /opt/arc/bin/nebius.new /opt/arc/bin/nebius && break
    sleep 5
  done
  [ -x /opt/arc/bin/nebius ] && echo "arc bootstrap: nebius cli $CLI_VERSION" \
    || echo "arc bootstrap: nebius cli $CLI_VERSION not installed, using the image's"
fi
[ -x /opt/arc/bin/nebius ] || command -v nebius >/dev/null \
  || curl -sSL https://artifacts.nebius.cloud/cli/install.sh | NEBIUS_INSTALL_FOLDER=/usr/local/bin bash \
  || fail "nebius cli"

# 3. Packages.
for attempt in 1 2 3; do
  apt-get -o DPkg::Lock::Timeout=300 update -q \
    && apt-get -o DPkg::Lock::Timeout=300 install -y -q --no-install-recommends $PACKAGES && break
  [ "$attempt" = 3 ] && fail "apt"
  sleep 10
done
# ninja: the Nebius GPU image pins the apt package away (01-ubuntu-conflicts-lock); the PyPI wheel carries the binary.
command -v ninja >/dev/null || { python3 -m venv /opt/arc/tools && /opt/arc/tools/bin/pip install -q ninja \
  && ln -sf /opt/arc/tools/bin/ninja /usr/local/bin/ninja && ninja --version; } || fail "ninja"

# 4. Bucket key from SecretStash: tmpfs and process environment only.
umask 077
/usr/local/bin/arc-nebius mysterybox payload get --secret-id "$SECRET_ID" --format json >/run/arc/payload.json \
  || fail "secret payload"
python3 - /run/arc/payload.json >/run/arc/s3.env <<'EOF' || fail "secret keys"
import json, sys
found = {}
def walk(node):
    if isinstance(node, dict):
        if isinstance(node.get("key"), str) and isinstance(node.get("string_value"), str):
            found[node["key"]] = node["string_value"]
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(json.load(open(sys.argv[1])))
for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    print(f"{name}={found[name]}")
EOF
rm -f /run/arc/payload.json
cat >>/run/arc/s3.env <<EOF
RCLONE_CONFIG_ARC_TYPE=s3
RCLONE_CONFIG_ARC_PROVIDER=Other
RCLONE_CONFIG_ARC_ENV_AUTH=true
RCLONE_CONFIG_ARC_ENDPOINT=$ENDPOINT
RCLONE_CONFIG_ARC_REGION=$REGION
RCLONE_CONFIG_ARC_NO_CHECK_BUCKET=true
HOME=/root
EOF
set -a; . /run/arc/s3.env; set +a
umask 022

# 5. This run's runner, bundle and spec; the runner is pinned by the launcher's hash.
rclone copy "arc:$BUCKET/$LAUNCH_PREFIX" /opt/arc/launch --retries 5 || fail "launch files"
echo "$RUNNER_SHA256  /opt/arc/launch/nebius_runner.py" | sha256sum -c - || fail "runner hash"

# 6. The run as its own unit: the preemption SIGTERM reaches the runner alone, and the VM deletes itself whenever
#    the unit ends (success, failure, crash).
cat >/etc/systemd/system/arc-run.service <<'EOF'
[Unit]
Description=ARC validation run
After=network-online.target
[Service]
Type=simple
EnvironmentFile=/run/arc/s3.env
ExecStart=/usr/bin/python3 -u /opt/arc/launch/nebius_runner.py /opt/arc/launch.json
ExecStopPost=/bin/sh -c 'systemd-run --unit arc-delete --no-block /usr/local/bin/arc-self-delete || /usr/local/bin/arc-self-delete'
KillMode=mixed
TimeoutStopSec=50
OOMPolicy=continue
LimitNOFILE=1048576
TasksMax=infinity
StandardOutput=append:/var/log/arc/runner.log
StandardError=inherit
EOF
systemctl daemon-reload && systemctl start --no-block arc-run.service || fail "runner unit"
"""


# -- configuration and pure helpers ------------------------------------------------------------------------------
def load_config(path=None):
    path = Path(path or os.environ.get("ARC_NEBIUS_CONFIG", "~/.nebius/arc.json")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; `nebius_validation.py setup` prints the template")
    return check_config(json.loads(path.read_text()))


def check_config(raw):
    config = {**DEFAULTS, **raw}
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        raise ValueError(f"Nebius config lacks {missing}")
    if config["region"] not in PLATFORMS:
        raise ValueError(f"RTX PRO 6000 is offered in {sorted(PLATFORMS)}, not {config['region']}")
    config.setdefault("platform", PLATFORMS[config["region"]])
    config.setdefault("endpoint", f"https://storage.{config['region']}.nebius.cloud")
    if config["boot_disk_type"] != "network_ssd" and config["boot_disk_gib"] % 93:
        raise ValueError("Non-replicated and IO M3 disks come in multiples of 93 GiB")
    if config["boot_disk_type"] not in PRICES["disk_gib_month"]:
        raise ValueError(f"unknown disk type {config['boot_disk_type']}")
    if bool(config["pricing_policy_id"]) != (config["spot_max_price_usd_per_gpu_hour"] is not None):
        raise ValueError("pricing_policy_id and spot_max_price_usd_per_gpu_hour go together")
    return config


def hourly_rate(config, *, preemptible=True):
    """USD per hour a bound is computed with, and where the GPU figure comes from."""
    if not preemptible:
        gpu, basis = PRICES["gpu_hour"]["on_demand"], "on-demand list price"
    elif not config["pricing_policy_id"] and not config["follow_spot_price"]:
        raise ValueError("A preemptible launch needs a pricing policy (max spot price) or follow_spot_price: true")
    elif config["pricing_policy_id"]:
        gpu = float(config["spot_max_price_usd_per_gpu_hour"])
        basis = f"pricing policy {config['pricing_policy_id']} max spot price"
        if not 0 < gpu <= PRICES["gpu_hour"]["on_demand"]:
            raise ValueError("The max spot price must be positive and at most the on-demand price")
    else:
        gpu, basis = PRICES["gpu_hour"]["on_demand"], "follows spot price: bounded by the on-demand list price"
    disk = config["boot_disk_gib"] * PRICES["disk_gib_month"][config["boot_disk_type"]] / 730
    return {"gpu_usd_per_hour": gpu, "disk_usd_per_hour": round(disk, 6), "usd_per_hour": gpu + disk, "basis": basis}


# The boot disk holds the image (~95 GiB used before any asset on 13 Oct), the assets and the run's output. AR v1-v3
# (13 Oct) fetched 342 GiB of assets onto the 372 GiB disk and the VM vanished when it filled.
# 17 Oct (cut provisioned disk under the 5 TiB quota): measured peak use = image ~38 GiB + fetched assets
# (nebius-diag df: AV2C family 379 GiB with 342 GiB of assets, GI3 273 with 236, DKb 211 with 173), plus a 20 GiB output
# allowance; size = that + 15%, in whole steps (no longer floored at the config's 372).
DISK_IMAGE_GIB, DISK_OUTPUT_GIB, DISK_MARGIN = 38, 20, 1.15


def asset_skip_for(study, assets):
    """``nebius_asset_skip`` ({prefix: [fnmatch patterns on manifest paths]}): files of a fetched asset the run never
    reads (20 Oct: the base expert shards an expert overlay replaces), so the runner neither fetches nor verifies
    them. Opt-in per study; a pattern that matches nothing is an error at the runner."""
    skip = study.get("nebius_asset_skip") or {}
    if not isinstance(skip, dict):
        raise ValueError("nebius_asset_skip must map an asset prefix to a list of patterns")
    for prefix, patterns in skip.items():
        if prefix not in assets:
            raise ValueError(f"nebius_asset_skip names {prefix!r}, which the run does not fetch")
        if not (isinstance(patterns, list) and patterns and all(isinstance(x, str) and x for x in patterns)):
            raise ValueError(f"nebius_asset_skip[{prefix!r}] must be a non-empty list of patterns")
    return {prefix: list(patterns) for prefix, patterns in skip.items()}


LATE_MAX_SECONDS = 4 * 3600


def asset_late_for(study, assets):
    """``nebius_asset_late`` ({prefix: {"wait_seconds": N, "gib": G}}): assets not in the bucket yet at
    launch (a checkpoint export still running). The VM fetches everything else first, then waits for the asset's
    manifest (the export writes it last) until ``wait_seconds`` after the VM's start, then fetches and checks it as
    usual. ``gib`` sizes the boot disk for it (its manifest is not there to read); the wait is reserved on top of the
    setup allowance. Opt-in per study; """
    late = study.get("nebius_asset_late") or {}
    if not isinstance(late, dict):
        raise ValueError("nebius_asset_late must map an asset prefix to {wait_seconds, gib}")
    out = {}
    for prefix, rule in late.items():
        if prefix not in assets:
            raise ValueError(f"nebius_asset_late names {prefix!r}, which the run does not fetch")
        wait, gib = (rule or {}).get("wait_seconds"), (rule or {}).get("gib")
        if not (isinstance(wait, int) and 60 <= wait <= LATE_MAX_SECONDS):
            raise ValueError(f"nebius_asset_late[{prefix!r}].wait_seconds must be an int in 60..{LATE_MAX_SECONDS}")
        if not (isinstance(gib, (int, float)) and gib > 0):
            raise ValueError(f"nebius_asset_late[{prefix!r}].gib must be the asset's size in GiB (> 0)")
        out[prefix] = {"wait_seconds": wait, "gib": gib}
    return out


def skipped(rel, patterns):
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns)


# Non-replicated disk bandwidth grows per 93 GiB unit (Nebius docs: 82 MiB/s write, 110 MiB/s read per unit, 2 GiB/s
# cap at 25 units); the asset fetch and a cold weight load run at that rate (20 Oct). ``nebius_boot_disk_gib`` (study)
# sizes one run's disk for bandwidth, inside the project's 5 TiB disk quota.
def boot_disk_for(config, cloud, assets, skip=None, floor_gib=None, extra_gib=0, late=None):
    """The config's boot disk, or a larger one (whole 93 GiB steps on non-replicated disks) when the assets need it,
    or the study's ``floor_gib`` when that is larger still. A late asset counts its declared ``gib``."""
    late = late or {}
    total, skip = sum(int(rule["gib"] * 2 ** 30) for rule in late.values()), skip or {}
    for prefix in assets:
        if prefix in late:
            continue
        manifest = cloud.read_json(MANIFESTS_PREFIX, f"{asset_slug(prefix)}.json") if not cloud.dry_run else None
        if manifest:
            total += sum(int(meta.get("size") or 0) for rel, meta in (manifest.get("files") or {}).items()
                         if not skipped(rel, skip.get(prefix, [])))
    step = 93 if config["boot_disk_type"] != "network_ssd" else 1
    if floor_gib is not None and (int(floor_gib) <= 0 or int(floor_gib) % step):
        raise ValueError(f"nebius_boot_disk_gib {floor_gib} is not a positive multiple of {step}")
    if not total:  # no manifest read (dry run, unmirrored prefix): the config's size, as before
        return max(int(config["boot_disk_gib"]), int(floor_gib or 0))
    need = (DISK_IMAGE_GIB + total / 2 ** 30 + DISK_OUTPUT_GIB + extra_gib) * DISK_MARGIN
    gib = step * max(1, -(-int(need) // step))  # the smallest whole step above the need
    return max(gib, int(floor_gib or 0))


def bound_usd(seconds, usd_per_hour):
    """modal_validation.reserve's shape: (seconds + 60) x rate x 1.15."""
    return (seconds + 60) * usd_per_hour / 3600 * 1.15


def sibling(kind, test, path=None):
    """The sibling Modal launcher without importing Modal: its plain top-level functions and literal constants in one
    namespace (``TEST`` / ``KIND`` as launched), and its syntax tree."""
    import types

    # ARC_MODAL_VALIDATION: another branch's launcher (its tests), e.g. a worktree's scripts/modal_validation.py.
    tree = ast.parse(Path(path or os.environ.get("ARC_MODAL_VALIDATION") or HERE / "modal_validation.py").read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            body.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
                body.append(node)
            except ValueError:
                pass
    scope = {"json": json, "Path": Path, "os": os, "time": time, "TEST": test, "KIND": kind,
             "modal": types.SimpleNamespace(is_local=lambda: True)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "modal_validation.py", "exec"), scope)
    return scope, tree


def _guard_shaped(statements):
    """Only assignments to plain names, nested ifs of the same shape, raises and calls of the sibling's own
    ``validate_*`` functions: a check with no side effect."""
    for node in statements:
        if isinstance(node, ast.Raise):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            continue
        if isinstance(node, ast.If) and _guard_shaped(node.body) and _guard_shaped(node.orelse):
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "").startswith("validate_")):
            continue
        return False
    return True


def sibling_guards(tree, scope, test, kind, bundle, run_id):
    """Run the Modal container's per-test bundle checks (``if TEST == ...: ... raise`` or ``validate_*(...)``)
    against the local bundle. Only blocks reached through conditions on TEST / KIND alone are considered: a check
    under any other condition (``if output.exists():``) is the container's business. Returns the number run and the
    ones skipped (they read container paths or names this side does not have)."""
    benchmark = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark"), None)
    found, ran, skipped = [], 0, []

    def visit(statements):
        for node in statements:
            if isinstance(node, (ast.With, ast.Try)):
                visit(node.body)
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if names - {"TEST", "KIND"}:
                continue  # not decidable here: nothing below it is run
            if not eval(compile(ast.Expression(node.test), "modal_validation.py", "eval"), {"TEST": test, "KIND": kind}):
                visit(node.orelse)
            elif "TEST" in names and _guard_shaped(node.body) and any(
                    isinstance(n, ast.Raise) or (isinstance(n, ast.Call) and getattr(n.func, "id", "").startswith(
                        "validate_")) for n in ast.walk(node)):
                found.append(node)
            else:
                visit(node.body)

    visit(benchmark.body if benchmark else [])
    for node in found:
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith(("/assets", "/runs", "/kaggle", "/tmp", "/opt")) for n in ast.walk(node)):
            skipped.append(f"line {node.lineno}: container path")
            continue
        local = {**scope, "source": Path(bundle), "run_id": run_id}
        try:
            exec(compile(ast.Module(body=node.body, type_ignores=[]), "modal_validation.py", "exec"), local)
            ran += 1
        except (ValueError, RuntimeError):
            raise
        except OSError as exc:
            if str(bundle) not in str(exc):
                skipped.append(f"line {node.lineno}: {type(exc).__name__}: {exc}")
                continue
            # The check reads a bundle file that is not there: not a bundle for this test.
            raise ValueError(f"{test}: the bundle lacks what this test checks ({exc})") from exc
        except Exception as exc:  # noqa: BLE001 - a check this side cannot evaluate is reported, not enforced
            skipped.append(f"line {node.lineno}: {type(exc).__name__}: {exc}")
    return ran, skipped


def plan_run(kind, test, study, run_id, bundle=None, path=None):
    """Timeout, bootstrap script, arguments, stop rule and bundle checks, as the sibling modal_validation.py chooses
    them for this test (whatever tests that branch's launcher defines); B200 tests fall back to the B200 rule."""
    if kind != "arena":
        raise ValueError("The Nebius backend runs arena bundles only")
    if study.get("backend") == "dfr-exact" or (study.get("dfr_exact") or {}).get("entry"):
        # dfranzen's public notebook with his own serving stack: the bundle's bootstrap_dfr.py is
        # the job, the study names its timeout and its asset prefixes; no sibling launcher, no vLLM base assets.
        # 12 October: a study may name another entry (``dfr_exact.entry``; v8.2's harness on his serving).
        block = study["dfr_exact"]
        entry = block.get("entry", "bootstrap_dfr.py")
        if not (bundle is None or (Path(bundle) / entry).is_file()):
            raise ValueError(f"a dfr-exact study needs {entry} in its bundle")
        return {"timeout": int(block["timeout_seconds"]), "script": entry, "arguments": [],
                "stop_rule": None, "assets": [str(a).strip("/") for a in block["assets"]],
                "play_seconds": int(block["play_backstop_seconds"])}
    if study.get("backend") == "image":
        # dots3 / mmrtx (24 Oct): a digest-pinned docker image runs the bundle's entry script on the GPU with the
        # VM's network; the study names the image, entry, timeout and asset prefixes (``image-bundle`` writes it).
        block = study["image"]
        ref = str(block["ref"])
        if "@sha256:" not in ref and not re.search(r":v?\d+\.\d+[\w.-]*$", ref):
            raise ValueError(f"image {ref!r}: pin a digest (@sha256:...) or a version tag")
        entry = block["entry"]
        if not (bundle is None or (Path(bundle) / entry).is_file()):
            raise ValueError(f"an image study needs {entry} in its bundle")
        assets = [str(a).strip("/") for a in block.get("assets") or []]
        bad = [a for a in assets if not re.fullmatch(r"[\w.-]+(/[\w.-]+)*", a) or ".." in a]
        if bad:
            raise ValueError(f"image assets are bucket prefixes under assets/: {bad}")
        return {"timeout": int(block["timeout_seconds"]), "script": entry, "arguments": list(block.get("arguments") or []),
                "stop_rule": None, "assets": assets, "play_seconds": int(block["timeout_seconds"]), "image": ref,
                "image_env": {str(k): str(v) for k, v in (block.get("env") or {}).items()},
                "image_disk_gib": int(block.get("image_disk_gib", 40))}
    if study.get("backend", "vllm") != "vllm" or study.get("teacher"):  # bundled games allowed
        raise ValueError("Nebius backend: vLLM arena studies without teacher access only")
    if test.startswith("expert-quant-calibrate"):
        raise ValueError("This test writes to the assets volume; run it on Modal")
    block = study.get("b200") or {}
    if test in B200_TESTS:
        if not run_id.startswith(("b2-", "b2d-", "b2o-")) or block.get("test") != test:
            raise ValueError(f"B200 runs use b2- run ids and a bundle built for {test}")
        if block.get("gpu", "B200") != "RTX-PRO-6000":
            raise ValueError("The Nebius backend launches RTX PRO 6000 studies only")
    scope, tree = sibling(kind, test, path)
    saved = os.environ.get("ARC_VALIDATION_BUNDLE")
    if bundle is not None:
        os.environ["ARC_VALIDATION_BUNDLE"] = str(bundle)  # a sibling may size a test from the bundle's study
    try:
        try:
            timeout, arguments = int(scope["validation_timeout"](kind, test)), list(scope["arena_arguments"](test))
        except (KeyError, ValueError) as exc:
            if test not in B200_TESTS:
                raise ValueError(f"Unsupported arena test: {test} (scripts/modal_validation.py does not plan it: "
                                 f"{type(exc).__name__}: {exc})") from exc
            timeout = 1740 if test == "b200-smoke" else min(86400, int(study["notebook_seconds"]) + 300)
            arguments = ["--trial", "duck-transfer"] if block["harness"] == "arena" else []
        if (study.get("chain") or {}).get("trials") and "--smoke" not in arguments:
            # Chain mode (warm servers, 20 Oct): one VM runs these trials back to back on one server.
            arguments = [x for name in study["chain"]["trials"] for x in ("--trial", name)]
        named = [arguments[i + 1] for i, a in enumerate(arguments[:-1]) if a == "--trial"]
        absent = [n for n in named if n not in [t.get("name") for t in study.get("trials") or []]]
        if absent and study.get("trials"):  # the bootstrap would refuse it on the VM, after the setup was paid for
            raise ValueError(f"{test} runs the trial {absent} and this bundle's study defines "
                             f"{[t.get('name') for t in study['trials']]}: the wrong test for this bundle?")
        scope["TIMEOUT"] = timeout
        guards = sibling_guards(tree, scope, test, kind, bundle, run_id) if bundle is not None else (0, [])
    finally:
        if bundle is not None:
            os.environ.pop("ARC_VALIDATION_BUNDLE") if saved is None else os.environ.update(ARC_VALIDATION_BUNDLE=saved)
    script = "b200_harness.py" if test in B200_TESTS and block.get("harness") != "arena" else "bootstrap_arena.py"
    plan = {"timeout": timeout, "script": script, "arguments": arguments,
            "stop_rule": block.get("stop") if test in B200_TESTS else None}
    if guards[0] or guards[1]:
        plan["guards"] = {"ran": guards[0], "skipped": guards[1]}
    return plan


def bake_spec(study):
    """Opt-in boot images (21 Oct; off unless the study asks). ``nebius_bake: true``: the VM fetches and verifies the
    study's assets, records them in /opt/arc/baked-assets.json, powers off and keeps its disk (``bake-image`` then
    makes the image and deletes the VM). ``nebius_image_id``: boot from such an image; the runner skips the fetch and
    sha256 of every asset the image's record holds with the same manifest and skip list (sizes still checked)."""
    if study.get("nebius_bake") and study.get("nebius_image_id"):
        raise ValueError("a bake starts from the stock image: nebius_bake and nebius_image_id together")
    if study.get("nebius_bake"):
        return {"mode": "bake"}
    image = study.get("nebius_image_id")
    if image is not None and not (isinstance(image, str) and image.startswith("computeimage-")):
        raise ValueError(f"nebius_image_id must be a computeimage- id, not {image!r}")
    return {"image_id": image, "use_baked": True} if image else {}


def launched_trials(study, arguments):
    wanted = {arguments[i + 1] for i, a in enumerate(arguments[:-1]) if a == "--trial"}
    return [t for t in study.get("trials") or [] if not wanted or t.get("name") in wanted]


def play_seconds(study, arguments, timeout):
    """Play-clock seconds of this launch: each launched trial's wall (the preview's on a competition bundle, an
    unconditional stop's if earlier); the timeout for trials that size themselves or state no wall."""
    trials, total = launched_trials(study, arguments), 0.0
    stop = study.get("early_stop") or {}
    for trial in trials:
        competition = study.get("competition") if trial.get("competition", True) else None
        if "--smoke" in arguments:
            wall = (study.get("smoke") or {}).get("wall_seconds")
        elif study.get("chain") and trial.get("wall_seconds") is not None:
            wall = trial["wall_seconds"]  # chain trials each carry their own wall
        elif competition and competition.get("preview_seconds") is not None:
            wall = competition["preview_seconds"]
        else:
            wall = trial.get("wall_seconds", study.get("wall_seconds"))
        if trial.get("kind") in SELF_SIZED or wall is None:
            return int(timeout)
        total += min(float(wall), float(stop["at_seconds"])) if stop.get("always") else float(wall)
    return int(min(total, timeout)) if trials else int(timeout)


def resumable(bundle, study, arguments):
    """``(checkpoints, all)``: the bundle carries the arena checkpoint code, and every trial this launch runs can be
    cut and continued (arena.checkpoint.from_env: Duck-transfer without the study features it cannot cut)."""
    source = Path(bundle) / "src/arena/checkpoint.py"
    if not source.is_file():
        return False, False
    unsupported = ("explorer", "parallel_samples", "llm_backlog", "cross_game_notes", "memory_block")
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "UNSUPPORTED":
            unsupported = ast.literal_eval(node.value)
    trials = launched_trials(study, arguments)
    return True, bool(trials) and "--smoke" not in arguments and not any(study.get(k) for k in unsupported) and all(
        t.get("kind") == "duck-transfer" for t in trials)


def reserved_seconds(attempt_seconds, timeout, config, *, checkpointing, max_attempts):
    """The whole run's reservation: one full attempt plus, per further attempt, a relaunch's setup and the work a
    preemption loses (a checkpoint interval when the bundle resumes, the whole run when it restarts from zero)."""
    lost = config["checkpoint_seconds"] + 120 if checkpointing else timeout
    overhead = config["relaunch_setup_seconds"] + lost
    return attempt_seconds + (max_attempts - 1) * overhead, overhead


def required_assets(study):
    """Asset prefixes (relative to the assets volume root) a study reads: the base set plus its overlay datasets."""
    datasets = list(study.get("model_datasets") or [])

    def walk(node):
        if isinstance(node, dict):
            value = node.get("dataset")
            if isinstance(value, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", value):
                datasets.append(value)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(study)
    extra = [f"flash-input/datasets/{name}" for name in dict.fromkeys(datasets)]
    # Whole research checkpoints already mirrored under an asset prefix (13 Oct: AutoRound W4A16 for the serving
    # ablation): the study names the prefixes it reads under /assets.
    for prefix in study.get("nebius_assets") or []:
        if not (isinstance(prefix, str) and re.fullmatch(r"flash-input/(models|datasets)/[\w./-]+", prefix)):
            raise ValueError(f"nebius_assets entry is not a flash-input prefix: {prefix!r}")
        extra.append(prefix.strip("/"))
    return [*BASE_ASSETS, *(p for p in extra if p not in BASE_ASSETS)]


def exchange_prefixes(study):
    """Study ``nebius_exchange_in``: folders of other runs' outputs (``<run id>/<path>``, under runs/validation) the
    runner copies into this run's ``exchange-in/`` every sync (14 Oct, RSI: loop and scoring VMs trade jobs and
    results through the bucket; the trial itself has no network)."""
    out = []
    for prefix in study.get("nebius_exchange_in") or []:
        if not (isinstance(prefix, str) and re.fullmatch(r"[\w.-]+(/[\w.-]+)*", prefix) and ".." not in prefix):
            raise ValueError(f"nebius_exchange_in entry is not a run-relative folder: {prefix!r}")
        out.append(prefix)
    return out


def asset_slug(prefix):
    return prefix.strip("/").replace("/", "--")


def slug(text, limit=63):
    value = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return value[:limit].rstrip("-")


def attempt_run_id(base, attempt):
    return base if attempt == 1 else f"{base}-a{attempt}"


def split_attempt(run_id):
    match = re.fullmatch(r"(.+)-a(\d+)", run_id)
    return (match.group(1), int(match.group(2))) if match else (run_id, 1)


def verify_bundle(bundle):
    identity = json.loads((bundle / "MANIFEST.json").read_text())
    if not identity["dependencies_prepared"]:
        raise RuntimeError("Bundle lacks offline dependencies; rebuild with --dependencies")
    for name, expected in identity["files"].items():
        path = bundle / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle file differs from its manifest: {name}")
    return identity


def pack_bundle(bundle, target):
    """The bundle as modal_validation ships it (no dataset-upload, notebooks or source.zip)."""
    def keep(info):
        parts = Path(info.name).parts
        if any(part in BUNDLE_IGNORE for part in parts) or info.name.endswith(".ipynb") or "__pycache__" in parts:
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(target, "w:gz", compresslevel=1) as tar:
        tar.add(bundle, arcname=".", filter=keep)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def render_bootstrap(spec, config):
    values = {"RUN_ID": spec["run_id"], "BUCKET": spec["bucket"], "ENDPOINT": config["endpoint"],
              "REGION": config["region"], "SECRET_ID": config["s3_secret_id"], "PROJECT_ID": config["project_id"],
              "LAUNCH_PREFIX": spec["launch_prefix"], "RUNS_PREFIX": spec["runs_prefix"],
              "DEADLINE_EPOCH": str(int(spec["deadline_epoch"])), "RUNNER_SHA256": spec["runner_sha256"],
              "PACKAGES": " ".join(spec.get("packages", PACKAGES)),
              "CLI_VERSION": config.get("cli_version") or ""}
    script = BOOTSTRAP
    for name, value in values.items():
        script = script.replace(f"@{name}@", shlex.quote(value))
    if re.search(r"@[A-Z_0-9]+@", script):
        raise ValueError("unrendered bootstrap placeholder")
    return script


def render_user_data(spec, config):
    """cloud-config: the bootstrap and the run spec as files, one command. No secrets."""
    def b64(text):
        return base64.b64encode(text.encode()).decode()

    lines = ["#cloud-config"]
    if config.get("ssh_user") and config.get("ssh_public_key"):
        lines += ["users:", f"  - name: {config['ssh_user']}", "    sudo: ALL=(ALL) NOPASSWD:ALL",
                  "    shell: /bin/bash", "    ssh_authorized_keys:", f"      - {config['ssh_public_key']}"]
    lines += ["write_files:",
              "  - path: /opt/arc/bootstrap.sh", "    permissions: '0700'", "    encoding: b64",
              f"    content: {b64(render_bootstrap(spec, config))}",
              "  - path: /opt/arc/launch.json", "    permissions: '0600'", "    encoding: b64",
              f"    content: {b64(json.dumps(spec, indent=1))}",
              "runcmd:", "  - [bash, /opt/arc/bootstrap.sh]", ""]
    return "\n".join(lines)


def create_command(config, name, labels, user_data, *, preemptible=True, platform=None, preset=None, disk=None,
                   image_family=None, image_id=None):
    disk_type, disk_gib = disk or (config["boot_disk_type"], config["boot_disk_gib"])
    argv = ["compute", "instance", "create", "--parent-id", config["project_id"], "--name", name,
            "--resources-platform", platform or config["platform"], "--resources-preset", preset or config["preset"],
            "--boot-disk-attach-mode", "READ_WRITE", "--boot-disk-managed-disk-name", f"{name}-boot",
            "--boot-disk-managed-disk-type", disk_type, "--boot-disk-managed-disk-size-gibibytes", str(disk_gib),
            *(["--boot-disk-managed-disk-source-image-id", image_id] if image_id else  # a baked image (opt-in)
              ["--boot-disk-managed-disk-source-image-family-image-family", image_family or config["image_family"]]),
            "--network-interfaces", json.dumps([{"name": "eth0", "ip_address": {},
                                                 **({"public_ip_address": {}} if config["public_ip"] else {}),
                                                 "subnet_id": config["subnet_id"]}]),
            "--service-account-id", config["service_account_id"],
            "--labels", ",".join(f"{k}={v}" for k, v in labels.items()), "--recovery-policy", "fail"]
    if preemptible:
        argv += ["--preemptible-on-preemption", "stop"]
        argv += (["--spot-pricing-policy-id", config["pricing_policy_id"]] if config["pricing_policy_id"]
                 else ["--follows-spot-price"])
    else:
        argv += ["--on-demand"]
    return [*argv, "--cloud-init-user-data", user_data]


def classify(instance_state, status, final, reaped, now, deadline_epoch):
    """One word for a run from the instance state and the bucket markers (see the docstring's preemption note)."""
    if final:
        state = final.get("state", "failed")
        if state == "interrupted":
            return "reaped" if reaped else "preempted"
        return state
    state = (instance_state or "").upper()
    if not state:
        return "reaped" if reaped else "lost"  # gone without a final marker
    if state in ("STOPPED", "STOPPING", "ERROR"):
        return "reaped" if reaped else "preempted"  # the runner never stops the VM without a marker
    if deadline_epoch and now > deadline_epoch + 300:
        return "overdue"
    return "running" if status else "starting"


def bake_kept(deadline, instance_state, final, now):
    """A finished bake keeps its VM and disk for `bake-image` (which deletes them) while it powers off and after:
    final "baked", VM not deleting/failed (RUNNING in the seconds before the poweroff, STOPPING, STOPPED), up to
    BAKE_KEEP_SECONDS past its deadline. 30 Oct: a STOPPING bake was deleted by `status` (the reconcile loop)."""
    if (final or {}).get("state") != "baked" or (instance_state or "").upper() in ("DELETING", "ERROR", ""):
        return False
    try:
        return now <= int(deadline) + BAKE_KEEP_SECONDS
    except (TypeError, ValueError):
        return False


def reap_reason(labels, instance_state, final, now):
    """Why a labelled VM should be deleted, or None."""
    state = (instance_state or "").upper()
    if bake_kept(labels.get("arc-deadline"), state, final, now):
        return None
    if state in ("STOPPED", "ERROR"):
        return f"instance {state.lower()} (preempted or powered off)"
    if final:
        return f"run finished ({final.get('state')}) but the VM is still there"
    try:
        deadline = int(labels.get("arc-deadline", ""))
    except ValueError:
        return "no deadline label"
    if now > deadline:
        return f"{int(now - deadline)} s past its deadline"
    return None


# -- ledger ------------------------------------------------------------------------------------------------------
@contextmanager
def ledger(root):
    """docs/spend.json under modal_validation.reserve's lock and formatting; nothing is written if the body raises."""
    with (Path(root) / "docs/spend.json").open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        yield data
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()


def set_cap(root, cap_usd, authorization):
    if not authorization.strip():
        raise ValueError("The cap needs the user's authorization text")
    with ledger(root) as data:
        book = data.setdefault("nebius", {"cap_usd": None, "cap_authorizations": [], "runs": [],
                                          "pricing_checked": PRICES["checked"], "pricing_url": PRICES["url"],
                                          "note": "Separate from the Modal cap_usd; not summed into it."})
        book["cap_authorizations"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           "previous_cap_usd": book["cap_usd"], "cap_usd": cap_usd,
                                           "authorization": authorization})
        book["cap_usd"] = cap_usd


def _records(root):
    """arc_infra.runops.records from the main checkout when the root keeps run records (docs/budget.json), else None
    (tests, old checkouts: the legacy in-place ledger)."""
    if not (Path(root) / "docs/budget.json").is_file():
        return None
    if str(Path(root) / "src") not in sys.path:  # the main checkout's runops, also under a plain python3
        sys.path.insert(0, str(Path(root) / "src"))
    from arc_infra.runops import records

    return records


@contextlib.contextmanager
def allowance_gate(root, entry):
    """The launching arm's own allowance plus the global caps (arc_infra.runops.records.gate), held
    under the records gate lock through the ledger write; then the run's live queue row (records), so the next
    launch counts it at once. The arm: ARC_ARM, else the longest run_arms prefix in docs/html/arm-state.json.
    ARC_ALLOWANCE_EXEMPT=<owner-approved reason> skips the arm's GPU allowance (throughput's c100 300K reads); the
    global caps still apply. ARC_WHAT: the queue row's plain-language description. Skipped where the root has no
    docs/budget.json (tests, old checkouts)."""
    records = _records(root)
    if records is None:
        yield None
        return
    run = entry["run_id"]
    arm = os.environ.get("ARC_ARM") or records.arm_for(run, records.arm_rules(root))
    if not arm:
        raise RuntimeError(f"No arm for {run}: set ARC_ARM=<arm> (no run_arms prefix matches)")
    long_24h = float(entry.get("play_seconds") or 0) >= 23 * 3600
    exempt = os.environ.get("ARC_ALLOWANCE_EXEMPT")
    flags = {**({"allowance_exempt": exempt} if exempt else {}), **({"long_run_24h": True} if long_24h else {})}
    entry.update(arm=arm, **flags)
    gpu = "RTX PRO 6000" if entry.get("gpu") else ""
    with records._gate_lock(root):
        for w in records.gate(arm, gpu, 1, float(entry["charged_bound_usd"]), platform="nebius",
                              long_run_24h=long_24h, allowance_exempt=exempt, root=root) or []:
            print(f"GATE WARNING: {w}", file=sys.stderr)
        yield records
        now = time.time()

        def row(rec):
            rec["arm"] = arm
            rec.setdefault("queue", {"section": "live", "order": records._now_order(), "row": {
                "run": run, "arm": arm, "platform": "nebius", "gpu": gpu or "CPU", "state": "running",
                "start": time.strftime("%H:%M", time.gmtime(now)), "start_epoch": round(now),
                "what": os.environ.get("ARC_WHAT") or f"(launched by nebius_validation; {entry.get('kind') or 'run'})",
                **flags}})

        records.update(run, row, root)
    records.publish(root)


def _check_cap(data, entry):
    book = data.get("nebius") or {}
    cap = book.get("cap_usd")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)):
        raise RuntimeError("No Nebius spend cap: ask the user, then `nebius_validation.py set-cap USD "
                           "--authorization ...` (it is separate from the Modal cap)")
    taken = [r["run_id"] for r in book["runs"]] + [r["run_id"] for r in data.get("runs", [])]
    if entry["run_id"] in taken:
        raise ValueError("Run already reserved; inspect its receipt instead of relaunching")
    from arc_infra.runops.records import row_cost

    # open rows at their bound, settled rows at actual cost
    committed = sum(row_cost(r)[0] for r in book["runs"])
    if committed + entry["charged_bound_usd"] > cap:
        raise RuntimeError(f"Nebius spend cap: ${committed:.2f} + ${entry['charged_bound_usd']:.2f} required, "
                           f"cap ${cap:.2f}")
    return book


def reserve(root, entry):
    """The ledger row: into the run's record (docs/runs/<run>.json, views regenerated) where the root keeps records,
    else in place in docs/spend.json."""
    with allowance_gate(root, entry) as records:
        if records is None:
            with ledger(root) as data:
                _check_cap(data, entry)["runs"].append(entry)
            return entry["charged_bound_usd"]
        _check_cap(json.loads((Path(root) / "docs/spend.json").read_text()), entry)

        def add(rec):
            if any(e["block"] == "nebius" for e in rec.get("ledger") or []):
                raise ValueError("Run already reserved; inspect its receipt instead of relaunching")
            rec.setdefault("ledger", []).append({"block": "nebius", "order": records._now_order(), "row": dict(entry)})

        records.update(entry["run_id"], add, root)
    return entry["charged_bound_usd"]


END_CAUSE = {"completed": "completed", "complete": "completed", "succeeded": "completed", "success": "completed",
             "finished": "completed", "ok": "completed", "preempted": "preempt", "interrupted": "preempt"}


def settle_entry(root, run_id, state, observed_seconds, note=None, bound=None):
    """Mark a ledger entry with its outcome; the bound drops to the observed lifetime + 120 s (never rises)."""
    def settle(entry):
        if entry["state"] != "reserved":
            return entry
        observed = max(0.0, float(observed_seconds))
        new = min(entry["charged_bound_usd"],
                  bound_usd(observed + 60, entry["rate_usd_per_hour"]) if bound is None else bound)
        entry.update(state=state, observed_seconds=round(observed, 1),
                     previous_charged_bound_usd=entry["charged_bound_usd"], charged_bound_usd=new,
                     note=note or "Observed VM lifetime + 120 s at the reserved rate; not provider-metered")
        return entry

    records = _records(root)
    if records is None:
        with ledger(root) as data:
            return dict(settle(next(r for r in data["nebius"]["runs"] if r["run_id"] == run_id)))
    out = {}

    def fn(rec):
        rows = [e for e in rec.get("ledger") or [] if e["block"] == "nebius"]
        if not rows:  # StopIteration here escaped as a bare traceback (mmrtx-kl-grid-v1, 24 Oct)
            raise RuntimeError(f"{run_id}: no nebius ledger row in {records.runs_dir(root)}; "
                               "is ARC_VALIDATION_ROOT the main checkout (records live there)?")
        out.update(settle(rows[-1]["row"]))
        if rec.get("state") != "ended" or rec.get("end_cause") in (None, "unclassified"):
            # the record's lifecycle (records.settle); an arm's stated cause is kept, else the VM outcome's
            cause = END_CAUSE.get(str(state).lower(), "unclassified")
            rec.update(state="ended", end_cause=cause)
            rows[-1]["row"].setdefault("end_cause", cause)
            if rec.get("attempts"):
                rec["attempts"][-1].update(state="ended", end_cause=cause)

    records.update(run_id, fn, root)
    records.publish(root)
    return dict(out)


# -- cloud calls -------------------------------------------------------------------------------------------------
def nebius_binary():
    return shutil.which("nebius") or str(Path("~/.nebius/bin/nebius").expanduser())


class Cloud:
    """Every Nebius CLI and bucket call goes through here; --dry-run prints each instead of running it."""

    def __init__(self, config, *, dry_run=False, echo=print):
        self.config, self.dry_run, self.echo, self.calls = config, dry_run, echo, []

    def key(self, *parts):
        return f"{self.config['rclone_remote']}:{self.config['bucket']}/" + "/".join(p.strip("/") for p in parts)

    def run(self, argv, *, display=None, check=True, timeout=None):
        self.calls.append(display or argv)
        if self.dry_run:
            self.echo("DRY-RUN " + shlex.join(display or argv))
            return None
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if check and done.returncode:
            raise RuntimeError(f"{shlex.join((display or argv)[:5])} exited {done.returncode}: "
                               f"{explain(done.stderr)}")
        return done

    def nebius_argv(self, *args):
        profile = ["--profile", self.config["profile"]] if self.config["profile"] else []
        return [self.config.get("nebius_binary") or nebius_binary(), *profile, *args, "--format", "json"]

    def nebius(self, *args, display=None, check=True, timeout=None):
        done = self.run(self.nebius_argv(*args), check=check, timeout=timeout,
                        display=self.nebius_argv(*display) if display else None)
        if done is None or done.returncode or not done.stdout.strip():
            return None
        try:
            return json.loads(done.stdout)
        except ValueError:
            return None

    def cli_version(self):
        """This side's Nebius CLI version (a local call), or None."""
        if self.dry_run:
            return None
        done = self.run([self.config.get("nebius_binary") or nebius_binary(), "version"], check=False, timeout=60)
        found = re.search(r"\b\d+\.\d+\.\d+\b", (done.stdout or "") if done is not None else "")
        return found.group(0) if found else None

    def missing_create_flags(self):
        """Flags of this launcher's instance create that this side's CLI does not list (a local --help call); None
        when the help itself could not be read."""
        done = self.run([self.config.get("nebius_binary") or nebius_binary(), "compute", "instance", "create",
                         "--help"], check=False, timeout=120)
        text = re.sub(r"\x1b\[[0-9;]*m", "", (done.stdout or "") + (done.stderr or "")) if done is not None else ""
        if "--parent-id" not in text:
            return None
        wanted = {a for spot in (True, False) for policy in (self.config.get("pricing_policy_id") or "p", None)
                  for a in create_command({**self.config, "pricing_policy_id": policy}, "x", {"a": "b"}, "",
                                          preemptible=spot) if a.startswith("--")}
        return sorted(f for f in wanted if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text))

    def quotas(self):
        """``{name: {"name", "limit", "usage"}}`` for the project's allowances that state a limit in this region
        (read-only: `quotas quota-allowance list`). An allowance without a limit says nothing (seen 7 October) and
        is left out: the create decides. ``usage`` None: not reported."""
        if self.dry_run:
            return {}
        listing = self.nebius("quotas", "quota-allowance", "list", "--parent-id", self.config["project_id"],
                              "--all", check=False)
        found = {}
        for item in (listing or {}).get("items", []):
            name = str((item.get("metadata") or {}).get("name", "")).lower()
            region, limit = (item.get("spec") or {}).get("region"), (item.get("spec") or {}).get("limit")
            usage = (item.get("status") or {}).get("usage")
            if limit is None or (region and region != self.config["region"]):
                continue
            try:
                found[name] = {"name": name, "limit": int(limit), "usage": None if usage is None else int(usage)}
            except (TypeError, ValueError):
                continue
        return found

    def gpu_quota(self, preemptible, quotas=None):
        """The RTX PRO 6000 allowance for this kind of VM, or None when none is recognised."""
        found = [{**q, "usage": q["usage"] or 0} for name, q in (self.quotas() if quotas is None else quotas).items()
                 if "gpu" in name and "6000" in name]
        mine = [q for q in found if ("preempt" in q["name"] or "spot" in q["name"]) == preemptible]
        return (mine or [q for q in found if len(found) == 1] or [None])[0]

    def gpu_vms(self, items=None):
        """Our GPU VMs that exist now (controller and preflight VMs are CPU)."""
        return [item for item in (self.instances() if items is None else items)
                if (item["metadata"].get("labels") or {}).get("arc-test") not in ("controller", "preflight")
                and ((item.get("status") or {}).get("state") or "").upper() != "DELETING"]

    def rclone(self, *args, check=True, timeout=None):
        return self.run(["rclone", *args], check=check, timeout=timeout)

    def read_json(self, *parts):
        done = self.rclone("cat", self.key(*parts), check=False, timeout=120)
        if done is None or done.returncode or not done.stdout.strip():
            return None
        try:
            return json.loads(done.stdout)
        except ValueError:
            return None

    def put_text(self, text, *parts):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / Path(parts[-1]).name
            path.write_text(text)
            self.rclone("copyto", str(path), self.key(*parts))

    def instances(self):
        listing = self.nebius("compute", "instance", "list", "--parent-id", self.config["project_id"], "--all")
        return [item for item in (listing or {}).get("items", [])
                if "arc-run" in (item.get("metadata", {}).get("labels") or {})]

    def instances_named(self, name):
        return [item["metadata"]["id"] for item in self.instances() if item["metadata"].get("name") == name]

    def instance_state(self, instance_id):
        if not instance_id:
            return None
        got = self.nebius("compute", "instance", "get", "--id", instance_id, check=False)
        return ((got or {}).get("status") or {}).get("state")

    def delete(self, instance_id):
        """Delete a VM with its managed disk; loud unless it is gone (or going) afterwards."""
        if not instance_id:
            raise RuntimeError("No instance id on record: run `list`, then delete the VM by hand or `reap --yes`")
        done = self.run(self.nebius_argv("compute", "instance", "delete", "--id", instance_id, "--async"),
                        check=False)
        if done is not None and done.returncode:
            if (self.instance_state(instance_id) or "DELETING").upper() != "DELETING":
                raise RuntimeError(f"Could not delete {instance_id}: {done.stderr.strip()[-300:]}")


def instance_id_of(created):
    for candidate in ((created or {}).get("metadata", {}).get("id"), (created or {}).get("resource_id")):
        if candidate and candidate.startswith("computeinstance-"):
            return candidate
    return None


# -- records -----------------------------------------------------------------------------------------------------
def record_path(root, run_id):
    return Path(root) / os.environ.get("ARC_VALIDATION_RECORDS", RECORDS) / f"nebius-{run_id}.json"


def read_record(root, run_id):
    return json.loads(record_path(root, run_id).read_text())


def write_record(root, data):
    path = record_path(root, data["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# -- commands ----------------------------------------------------------------------------------------------------
def save_record(root, cloud, record):
    """The receipt: docs/validation-nebius (when this side has the repo) and launch/<run>/record.json in the bucket,
    which is what a controller VM works from."""
    record["updated_epoch"] = round(time.time(), 3)
    if root is not None:
        write_record(root, record)
    cloud.put_text(json.dumps(record, indent=1), "launch", record["run_id"], "record.json")


def load_record(root, cloud, run_id):
    """The newer of the local receipt and the bucket's (a controller may have relaunched since)."""
    local = read_record(root, run_id) if root is not None and record_path(root, run_id).is_file() else None
    remote = cloud.read_json("launch", run_id, "record.json")
    def age(record):
        return len(record.get("attempts") or []), record.get("updated_epoch", 0)

    if local is None or (remote and age(remote) > age(local)):
        if remote is None and local is None:
            raise FileNotFoundError(f"No receipt for {run_id}")
        return remote or local
    return local


class CreateFailed(RuntimeError):
    """The VM was not created and nothing of it is left. ``retry``: no capacity (or no answer): the run goes on."""

    def __init__(self, message, *, retry, removed):
        super().__init__(message)
        self.retry, self.removed = retry, removed


def start_attempt(config, cloud, record, kind, now):
    """Create the VM of the run's next attempt: ``resume`` continues the same output folder from its checkpoint,
    ``fresh`` starts from zero under <run>-a<N> (never in a folder that holds a partial run)."""
    attempts = record["attempts"]
    n = len(attempts) + 1
    if kind == "resume":
        run_id, segment = attempts[-1]["run_id"], attempts[-1]["segment"] + 1
    else:
        run_id, segment = attempt_run_id(record["run_id"], n), 1
    left = record["reserved_seconds"] - sum(a.get("observed_seconds", 0) for a in attempts)
    seconds = min(record["attempt_seconds"], left)
    setup = config["setup_allowance_seconds"] + FINALIZE_SECONDS
    if seconds - setup < 600:
        raise RuntimeError(f"reservation used: {left:.0f} s left of {record['reserved_seconds']} s")
    spec = {**record["spec"], "run_id": run_id, "attempt": segment, "launch_attempt": n,
            "timeout": int(min(record["spec"]["timeout"], seconds - setup)), "deadline_epoch": now + seconds}
    name = slug(f"arc-{record['run_id']}", 58) if n == 1 else f"{slug('arc-' + record['run_id'], 54)}-{n}"
    labels = {"arc-run": slug(record["run_id"]), "arc-deadline": str(int(now + seconds)),
              "arc-test": slug(record["test"]), "arc-attempt": str(n)}
    if segment > 1:  # the earlier machine's markers must not describe this one
        base = (RUNS_PREFIX, "validation", run_id)
        for marker in ("nebius/final.json", "nebius/status.json", "execution.json", "bootstrap.log"):
            cloud.rclone("moveto", cloud.key(*base, marker),
                         cloud.key(*base, f"nebius/attempts/{segment - 1}/{Path(marker).name}"), check=False,
                         timeout=120)
    user_data = render_user_data(spec, config)
    argv = create_command(config, name, labels, user_data, preemptible=record["preemptible"],
                          disk=(config["boot_disk_type"], record.get("boot_disk_gib") or config["boot_disk_gib"]),
                          image_id=record["spec"].get("image_id"))
    shown = argv
    if cloud.dry_run:
        kept = Path(tempfile.mkdtemp(prefix="arc-nebius-dry-")) / "user-data.yaml"
        kept.write_text(user_data)
        shown = [*argv[:-1], f"$(cat {kept})"]
    try:
        created = cloud.nebius(*argv, display=shown, timeout=config["create_timeout_seconds"])
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        # A create that fails for capacity leaves the instance STOPPED with its boot disk (billed while it exists),
        # or (14 Oct) STARTING with its delete queued behind the create for hours. The cancel marker goes first: a VM
        # that boots after all finds it and ends before the bootstrap (its unit then deletes it); reap retries.
        if not cloud.dry_run:
            cloud.put_text(json.dumps({"launch_attempt": n, "instance_name": name, "epoch": round(now, 1),
                                       "why": f"{type(exc).__name__}: {str(exc)[:300]}"}) + "\n",
                           RUNS_PREFIX, "validation", run_id, CANCEL_MARKER)
        removed = [] if cloud.dry_run else cloud.instances_named(name)
        for instance in removed:
            cloud.delete(instance)
        text = f"{type(exc).__name__}: {exc}" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
        raise CreateFailed(text[:1000], removed=removed, retry=isinstance(exc, subprocess.TimeoutExpired)
                           or bool(NO_CAPACITY.search(text))) from exc
    attempt = {"n": n, "run_id": run_id, "segment": segment, "kind": "resumed" if kind == "resume" else "fresh",
               "instance_name": name, "created_epoch": now, "deadline_epoch": now + seconds,
               "instance_id": None if cloud.dry_run else (instance_id_of(created)
                                                          or (cloud.instances_named(name) or [None])[0])}
    attempts.append(attempt)
    record.update(state="submitted", instance_name=name, instance_id=attempt["instance_id"], created_epoch=now,
                  deadline_epoch=now + seconds, attempt=n, pending=None,
                  outputs=f"s3://{config['bucket']}/{RUNS_PREFIX}/validation/{run_id}")
    return attempt


def launch(root, config, cloud, *, run_id, kind, test, bundle, preemptible=False, wait=True, authorization=None,
           now=None, max_attempts=None, approval=None, relaunch=True, reason=None, checkpoint=None):
    bundle = Path(bundle).resolve()
    study = json.loads((bundle / "configs/comparison.json").read_text())
    verify_bundle(bundle)
    plan = plan_run(kind, test, study, run_id, bundle)
    if not config.get("cli_version"):
        config["cli_version"] = cloud.cli_version()  # every VM of the run installs this side's CLI version
    play = int(plan.get("play_seconds") or play_seconds(study, plan["arguments"], plan["timeout"]))
    needs = ("approval" if play > APPROVAL_SECONDS and not approval else
             "reason" if LONG_RUN_SECONDS < play <= APPROVAL_SECONDS and not (reason or approval) else None)
    if needs and not cloud.dry_run:
        raise ValueError(f"{test} plays {play} s: " + (
            "more than 4 h of play needs the user's approval (--approval \"<the user's words>\")"
            if needs == "approval" else
            "more than 2 h of play needs --reason \"<why the mechanism needs it>\" (above 4 h: --approval)"))
    long_run = {**({"long_run_approval": approval} if approval and play > LONG_RUN_SECONDS else {}),
                **({"long_run_reason": reason} if reason and play > LONG_RUN_SECONDS else {})}
    # A bundle built from a branch with src/arena/checkpoint.py checkpoints and resumes its Duck-transfer trials;
    # anything else restarts from zero and is reserved for it.
    checkpointing, continues = resumable(bundle, study, plan["arguments"])
    if not (preemptible if checkpoint is None else checkpoint):
        checkpointing = False  # a fixed instance checkpoints only when asked to
    continues = continues and checkpointing
    attempts_allowed = max(1, int(max_attempts or config["max_attempts"])) if preemptible and relaunch else 1
    rate = hourly_rate(config, preemptible=preemptible)
    attempt_seconds = plan["timeout"] + config["setup_allowance_seconds"] + FINALIZE_SECONDS
    max_seconds, overhead = reserved_seconds(attempt_seconds, plan["timeout"], config, checkpointing=continues,
                                             max_attempts=attempts_allowed)
    bound = bound_usd(max_seconds, rate["usd_per_hour"])
    assets = plan.get("assets") or required_assets(study)
    asset_late = asset_late_for(study, assets)
    if asset_late:  # the wait for a late asset is setup time: reserved, and the deadline leaves the run its timeout
        attempt_seconds += max(rule["wait_seconds"] for rule in asset_late.values())
        max_seconds, overhead = reserved_seconds(attempt_seconds, plan["timeout"], config, checkpointing=continues,
                                                 max_attempts=attempts_allowed)
        bound = bound_usd(max_seconds, rate["usd_per_hour"])
    runner = HERE / "nebius_runner.py"
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    launch_prefix = f"launch/{run_id}"
    policy = {"auto_relaunch": attempts_allowed > 1, "max_attempts": attempts_allowed, "checkpointing": checkpointing,
              "heartbeat_seconds": config["heartbeat_seconds"],
              "create_timeout_seconds": config["create_timeout_seconds"],
              "every_trial_resumes": continues,
              "checkpoint_seconds": config["checkpoint_seconds"] if checkpointing else None,
              "relaunch_overhead_seconds": overhead, "capacity_retry_seconds": config["capacity_retry_seconds"],
              "rule": "relaunch after a preemption only: resume from the newest checkpoint in the same folder, or "
                      "from zero under <run>-a<N> when there is none; never after another failure; without spot "
                      "capacity for capacity_retry_seconds the run ends as relaunch_failed (no on-demand fallback)"
              if attempts_allowed > 1 else "one attempt on a fixed instance: no relaunch, no controller"
              if not preemptible else "one attempt: no relaunch"}
    ours = [] if cloud.dry_run else cloud.instances()
    quotas = cloud.quotas()
    in_use = None if cloud.dry_run else len(cloud.gpu_vms(ours))
    quota = cloud.gpu_quota(preemptible, quotas)
    address = dict(quotas[ADDRESSES]) if config["public_ip"] and ADDRESSES in quotas else None
    if address and address["usage"] is None:  # not reported: at least the VMs of ours that exist hold one each
        address.update(usage=len([i for i in ours if ((i.get("status") or {}).get("state") or "").upper()
                                  != "DELETING"]), usage_basis="our VMs")
    print(json.dumps({"run_id": run_id, "test": test, "timeout": plan["timeout"], "play_seconds": play,
                      "long_run": needs or long_run or None, "attempt_seconds": attempt_seconds,
                      "max_seconds": max_seconds, "rate": rate, "charged_bound_usd": round(bound, 4),
                      "assets": assets, "preemptible": preemptible, "policy": policy,
                      "gpu_vms_now": in_use, "quota": quota, "max_parallel_gpu_vms": config["max_parallel_gpu_vms"],
                      "public_ip": config["public_ip"], "public_addresses": address,
                      "guards": plan.get("guards")}))

    if quota and quota["usage"] + 1 > quota["limit"]:
        raise RuntimeError(f"Quota: {quota['name']} allows {quota['limit']} and {quota['usage']} are in use: no room "
                           f"for another RTX PRO 6000 VM ({in_use} are ours, see `list`). Wait for a run to end or "
                           "ask Nebius for more quota (console: Administration > Limits)")
    if address and address["usage"] + 1 > address["limit"]:
        raise RuntimeError(f"No public IPv4 address is free: {ADDRESSES} allows {address['limit']} and "
                           f"{address['usage']} are in use (every VM takes one; {in_use} GPU VMs are ours, see "
                           "`list`). Wait for a run to end, raise the quota in the console (Administration > "
                           "Limits), or set public_ip false in the config once `preflight` has passed with it")
    if config["max_parallel_gpu_vms"] is not None and in_use is not None and in_use >= config["max_parallel_gpu_vms"]:
        raise RuntimeError(f"{in_use} GPU VMs of ours exist and max_parallel_gpu_vms is "
                           f"{config['max_parallel_gpu_vms']} (config): wait for a run to end, see `list`")
    listing = cloud.rclone("lsf", cloud.key(MANIFESTS_PREFIX) + "/", check=False, timeout=120)
    if listing is not None:
        have = set(listing.stdout.split())
        missing = [p for p in assets if f"{asset_slug(p)}.json" not in have and p not in asset_late]
        if missing:
            raise RuntimeError(f"Assets not mirrored to the bucket: {missing}; see `mirror-plan`")
    asset_skip = asset_skip_for(study, assets)
    disk_gib = boot_disk_for(config, cloud, assets, asset_skip, study.get("nebius_boot_disk_gib"),
                             extra_gib=plan.get("image_disk_gib", 0), late=asset_late)  # an image run also holds its docker image
    if disk_gib != config["boot_disk_gib"]:
        rate = hourly_rate({**config, "boot_disk_gib": disk_gib}, preemptible=preemptible)
        bound = bound_usd(max_seconds, rate["usd_per_hour"])
    entry = {"run_id": run_id, "state": "reserved", "kind": kind, "platform": "nebius", "charged_bound_usd": bound,
             "max_seconds": max_seconds, "attempt_seconds": attempt_seconds, "max_attempts": attempts_allowed,
             "gpu": "RTX-PRO-6000", "nebius_platform": config["platform"],
             "preset": config["preset"], "region": config["region"], "preemptible": preemptible,
             "rate_usd_per_hour": rate["usd_per_hour"], "gpu_rate_usd_per_hour": rate["gpu_usd_per_hour"],
             "disk_rate_usd_per_hour": rate["disk_usd_per_hour"], "rate_basis": rate["basis"],
             "reservation": "the whole run: one full attempt + (max_attempts - 1) x (relaunch setup + lost work)"
             if attempts_allowed > 1 else "one attempt: timeout + setup allowance + finalize",
             "play_seconds": play, **long_run,
             "authorization": authorization or "see nebius.cap_authorizations"}
    if cloud.dry_run:
        cloud.echo(f"DRY-RUN ledger: reserve ${bound:.2f} for {run_id} in docs/spend.json nebius.runs "
                   f"(rate ${rate['usd_per_hour']:.4f}/h, {rate['basis']}; {max_seconds} s for up to "
                   f"{attempts_allowed} attempts)")
    else:
        reserve(root, entry)
    created_epoch = now or time.time()
    record = {"state": "submission_intent", "run_id": run_id, "kind": kind, "test": test, "platform": "nebius",
              "charged_bound_usd": bound, "platform_policy": "Nebius AI Cloud RTX PRO 6000, "
              + ("preemptible" if preemptible else "on-demand"), "bundle": str(bundle), "region": config["region"],
              "nebius_platform": config["platform"], "preset": config["preset"], "preemptible": preemptible,
              "rate": rate, "max_seconds": max_seconds, "reserved_seconds": max_seconds,
              "attempt_seconds": attempt_seconds, "policy": policy, "attempts": [],
              "first_created_epoch": created_epoch, "created_epoch": created_epoch,
              "deadline_epoch": created_epoch + attempt_seconds, "runner_sha256": runner_sha,
              "play_seconds": play, **long_run, "boot_disk_gib": disk_gib,
              "outputs": f"s3://{config['bucket']}/{RUNS_PREFIX}/validation/{run_id}"}
    if not cloud.dry_run:
        write_record(root, record)

    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "bundle.tar.gz"
        record["bundle_sha256"] = pack_bundle(bundle, archive)
        record["spec"] = {
            "mode": "run", "kind": kind, "test": test, "timeout": plan["timeout"],
            "script": plan["script"], "arguments": plan["arguments"], "stop_rule": plan["stop_rule"],
            "gpu": "RTX-PRO-6000", "bucket": config["bucket"], "runs_prefix": RUNS_PREFIX,
            "assets_prefix": ASSETS_PREFIX, "manifests_prefix": MANIFESTS_PREFIX, "launch_prefix": launch_prefix,
            "assets": assets, **({"asset_skip": asset_skip} if asset_skip else {}),
            **({"asset_late": asset_late} if asset_late else {}),
            "verify": config["verify"], "block_network": not plan.get("image"),
            **({"image": plan["image"], "image_env": plan["image_env"]} if plan.get("image") else {}),
            "finalize_allowance": FINALIZE_SECONDS, "sync_seconds": 60,
            "rate_per_second": rate["usd_per_hour"] / 3600, "bundle_sha256": record["bundle_sha256"],
            "runner_sha256": runner_sha, "image_family": config["image_family"],
            "checkpoint": {"seconds": config["checkpoint_seconds"]} if checkpointing else None,
            "exchange_in": exchange_prefixes(study), **bake_spec(study)}
        if not cloud.dry_run:
            write_record(root, record)  # with the spec: a launcher that dies in the create can be picked up
        try:
            cloud.rclone("copyto", str(archive), cloud.key(launch_prefix, "bundle.tar.gz"))
            cloud.rclone("copyto", str(runner), cloud.key(launch_prefix, "nebius_runner.py"))
            start_attempt(config, cloud, record, "fresh", created_epoch)
        except CreateFailed as exc:
            if not exc.retry or not preemptible:
                # A fixed instance has no watcher to retry for it: no capacity ends the launch, at no cost.
                why = ("no capacity" if exc.retry else "quota exceeded" if NO_QUOTA.search(str(exc))
                       else "instance create failed")
                settle_entry(root, run_id, "launch_failed_no_execution", 0, bound=0.0,
                             note=f"{why}, nothing left of it: {said(exc)}")
                record.update(state="launch_failed_no_execution", error=str(exc)[-600:], reason=why,
                              deleted_instances=exc.removed)
                write_record(root, record)
                hint = (" Every VM takes a public IPv4 address: wait for a run to end, raise the quota in the "
                        "console, or set public_ip false in the config once `preflight` has passed with it."
                        if ADDRESSES in str(exc) else "")
                raise RuntimeError(f"{run_id} was not started ({why}): no VM or disk is left and nothing is "
                                   f"charged. Nebius: {said(exc)}.{hint}") from None
            # No capacity is not a failure of the run: it has not begun. The reservation stays, the watcher retries.
            record.update(state="relaunch_pending", pending={
                "kind": "fresh", "since_epoch": created_epoch, "failures": 1, "reason": "no_capacity",
                "last_error": str(exc)[-300:]})
            save_record(root, cloud, record)
            print(json.dumps({"event": "no_capacity", "run_id": run_id, "state": "relaunch_pending",
                              "removed": exc.removed, "retry_for_seconds": config["capacity_retry_seconds"],
                              "error": str(exc)[-300:],
                              "next": "the controller VM or `watch` retries the create; nothing is charged"}))
            return record
        except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
            if not cloud.instances_named(slug(f"arc-{run_id}", 58)):  # nothing was created: release the reservation
                settle_entry(root, run_id, "launch_failed_no_execution", 0, bound=0.0,
                             note=f"upload or instance create failed, no VM exists: {str(exc)[-300:]}")
                record.update(state="launch_failed_no_execution", error=str(exc)[-600:])
                write_record(root, record)
            raise
    if cloud.dry_run:
        cloud.echo(f"DRY-RUN receipt: {record_path(root, run_id)} (state submitted, instance id from the create)")
        cloud.echo(f"DRY-RUN wait: poll {cloud.key(RUNS_PREFIX, 'validation', run_id, 'nebius/status.json')} for "
                   f"{config['heartbeat_seconds']} s, delete the VM if it never appears")
        return record
    save_record(root, cloud, record)
    print(json.dumps({k: v for k, v in record.items() if k != "spec"}))
    if wait:
        wait_for_heartbeat(root, config, cloud, record)
    if policy["auto_relaunch"] and not controller_alive(cloud):
        print("No controller VM is up: a preemption is relaunched only while `watch` runs here "
              "(`controller up` keeps relaunching with the laptop asleep).")
    return record


def wait_for_heartbeat(root, config, cloud, record, poll=20):
    """A VM that never reports cannot be trusted to delete itself: delete it from here."""
    run_id, end = record["run_id"], time.time() + config["heartbeat_seconds"]
    while time.time() < end:
        status = cloud.read_json(RUNS_PREFIX, "validation", run_id, "nebius/status.json")
        if status:
            print(json.dumps({"event": "heartbeat", "phase": status.get("phase"),
                              "seconds": round(time.time() - record["created_epoch"])}))
            return status
        if (cloud.instance_state(record.get("instance_id")) or "").upper() in ("STOPPED", "ERROR"):
            break
        time.sleep(poll)
    cloud.delete(record["instance_id"])
    settle_entry(root, run_id, "failed_no_heartbeat", time.time() - record["created_epoch"])
    record.update(state="failed_no_heartbeat")
    write_record(root, record)
    raise RuntimeError(f"No heartbeat from {record['instance_id']} in {config['heartbeat_seconds']} s; VM deleted")


def decide(record, state, final, checkpoint):
    """What follows an ended attempt: ``resume``, ``fresh`` or None (the run is over)."""
    policy = record.get("policy") or {}
    if not policy.get("auto_relaunch") or len(record["attempts"]) >= policy["max_attempts"]:
        return None
    if state == "preempted":
        saved = (checkpoint or {}).get("trials") or (checkpoint or {}).get("done")
        return "resume" if policy["checkpointing"] and saved else "fresh"
    if state == "failed" and (final or {}).get("resume_failed"):
        return "fresh"  # the checkpoint could not be continued: the run itself has not failed
    return None


def inspect(root, cloud, run_id, now=None):
    """The run's record, markers and state. An ended attempt is closed (its VM deleted) and either a relaunch is
    left pending or the run is settled in receipt and ledger. ``root`` None: the controller, bucket records only."""
    now = now or time.time()
    record = load_record(root, cloud, run_id)
    if record.get("state") == "submission_intent" and "attempts" in record and root is not None and not cloud.dry_run:
        # The launcher has not reported its create. Still inside the call's time limit: wait. Past it the launcher
        # is gone: whatever it left is deleted and the run is retried like any create that found no capacity.
        patience = (record.get("policy") or {}).get("create_timeout_seconds", 900) + 60
        if now - record["created_epoch"] <= patience:
            return {"run_id": run_id, "state": "scheduling", "instance_state": None, "phase": None, "final": None,
                    "attempt": 1, "record_state": "submission_intent"}, record
        removed = cloud.instances_named(slug(f"arc-{run_id}", 58))
        for instance in removed:
            cloud.delete(instance)
        if "spec" in record and record.get("preemptible"):
            record.update(state="relaunch_pending", pending={
                "kind": "fresh", "since_epoch": record["created_epoch"], "failures": 1, "reason": "no_capacity",
                "last_error": f"the launcher never reported its create; removed {removed}"})
            save_record(root, cloud, record)
        else:
            settle_entry(root, run_id, "launch_failed_no_execution", 0, bound=0.0,
                         note=f"the launcher never reported its create; removed {removed}; nothing ran")
            record.update(state="launch_failed_no_execution")
            write_record(root, record)
    current = (record.get("attempts") or [{}])[-1]
    base = (RUNS_PREFIX, "validation", current.get("run_id", run_id))
    final = cloud.read_json(*base, "nebius/final.json")
    status = cloud.read_json(*base, "nebius/status.json")
    reaped = cloud.read_json(*base, "nebius/reap.json")
    instance_state = cloud.instance_state(record.get("instance_id"))
    state = classify(instance_state, status, final, reaped, now, record.get("deadline_epoch"))
    if state == "preempted" and record.get("preemptible") is False:
        state = "stopped"  # a fixed instance is not preempted: it was stopped, or it failed without a marker
    beat = (record.get("policy") or {}).get("heartbeat_seconds")
    if state == "starting" and beat and current and now - record["created_epoch"] > beat + 120:
        # A relaunched VM nobody waits on: without a heartbeat it cannot be trusted to delete itself. One failed
        # bucket read must not kill a live VM (arc2 dirafter, 16 Oct): re-read before giving up.
        for _ in range(2):
            time.sleep(0 if cloud.dry_run else 5)
            status = cloud.read_json(*base, "nebius/status.json")
            if status is not None:
                state = classify(instance_state, status, final, reaped, now, record.get("deadline_epoch"))
                if state == "preempted" and record.get("preemptible") is False:
                    state = "stopped"
                break
        else:
            state, final = "failed", {"state": "failed", "error": f"no heartbeat in {beat + 120} s",
                                      "ended_epoch": now}
    info = {"run_id": run_id, "state": state, "instance_state": instance_state,
            "phase": (status or {}).get("phase"), "final": final, "attempt": record.get("attempt", 1)}
    if record.get("state") == "relaunch_pending":
        info["state"] = "relaunch_pending"
    elif state in TERMINAL and record.get("state") not in TERMINAL and not cloud.dry_run:
        ended = (final or {}).get("ended_epoch") or ((final or {}).get("interrupted") or {}).get("epoch") or now
        observed = ended - record["created_epoch"]
        if bake_kept(record.get("deadline_epoch"), instance_state, final, now):
            pass  # bake-image images its disk, then deletes it; reap takes a forgotten one 6 h past its deadline
        elif instance_state and instance_state.upper() != "DELETING":
            cloud.delete(record["instance_id"])  # stopped or finished but still there: disk charge, quota
        checkpoint = cloud.read_json(*base, "nebius/checkpoint.json")
        current.update(state=state, observed_seconds=round(observed, 1), final=final,
                       checkpoint=(checkpoint or {}).get("trials"))
        following = decide(record, state, final, checkpoint)
        if following:
            record.update(state="relaunch_pending", pending={"kind": following, "since_epoch": now, "failures": 0})
            info["state"] = "relaunch_pending"
        else:
            total = sum(a.get("observed_seconds", 0) for a in record.get("attempts") or [current])
            record.update(state=state, final=final, observed_seconds=round(total, 1))
            if root is not None:
                entry = settle_entry(root, run_id, state, total)
                record["charged_bound_usd"] = entry["charged_bound_usd"]
        save_record(root, cloud, record)
    elif root is not None and record.get("state") in TERMINAL and "attempts" in record:
        # A controller ended the run: the ledger is settled here, where the ledger is.
        entry = settle_entry(root, run_id, record["state"], record.get("observed_seconds", 0),
                             bound=None if record["attempts"] else 0.0)  # no VM ever ran: nothing to charge
        record["charged_bound_usd"] = entry["charged_bound_usd"]
        write_record(root, record)
    info["record_state"] = record.get("state")
    return info, record


def advance(root, config, cloud, run_id, now=None, fresh=False):
    """``inspect`` plus the pending relaunch (retried while spot capacity is missing). A local caller leaves the
    relaunch to a live controller VM."""
    now = now or time.time()
    info, record = inspect(root, cloud, run_id, now)
    pending = record.get("pending") if record.get("state") == "relaunch_pending" else None
    if pending:
        info["pending"] = pending
    # A live controller relaunches; this side steps in when it keeps failing for another reason than capacity (e.g.
    # its service account may not create).
    if not pending or cloud.dry_run or (root is not None and controller_alive(cloud, now) and (
            pending.get("failures", 0) < 3 or pending.get("reason") == "no_capacity")):
        return info, record
    # One create at a time per run: it can block for minutes, long enough for the other side to think this one gone.
    me = "controller" if root is None else "local"
    claim = cloud.read_json("launch", run_id, "claim.json") or {}
    if claim.get("by") not in (None, me) and now - claim.get("epoch", 0) < CLAIM_SECONDS:
        info["claimed_by"] = claim["by"]
        return info, record
    cloud.put_text(json.dumps({"by": me, "epoch": round(now, 1)}), "launch", run_id, "claim.json")
    try:
        try:
            attempt = start_attempt(config, cloud, record, "fresh" if fresh else pending["kind"], now)
        finally:  # the create has answered: the claim covers a call in flight only
            cloud.put_text(json.dumps({"by": None, "released_by": me}), "launch", run_id, "claim.json")
        info.update(state="relaunched", relaunched=attempt)
    except RuntimeError as exc:
        same = str(exc)[-300:] == pending.get("last_error") and not getattr(exc, "retry", False)
        pending.update(failures=pending["failures"] + 1, last_error=str(exc)[-300:],
                       reason="no_capacity" if getattr(exc, "retry", False) else "create_failed",
                       repeats=pending.get("repeats", 1) + 1 if same else 1)
        if "reservation used" in str(exc) or now - pending["since_epoch"] > config["capacity_retry_seconds"]:
            total = sum(a.get("observed_seconds", 0) for a in record["attempts"])
            record.update(state="relaunch_failed", observed_seconds=round(total, 1))
            if record.get("play_seconds", 0) > APPROVAL_SECONDS:
                record["next"] = "a run over 4 h: ask the user before it is started again"
            if root is not None:
                record["charged_bound_usd"] = settle_entry(
                    root, run_id, "relaunch_failed", total,
                    bound=None if record["attempts"] else 0.0)["charged_bound_usd"]
            info["state"] = "relaunch_failed"
    save_record(root, cloud, record)
    return info, record


def live_summary(cloud, run_id):
    progress = cloud.read_json(RUNS_PREFIX, "validation", run_id, "arena/live/progress.json")
    if not progress:
        return None
    try:
        sys.path.insert(0, str(HERE / "arena"))
        from watch_live import summary
        return summary(progress)
    except Exception:  # noqa: BLE001 - the Modal watcher's formatter is a convenience here
        live = progress.get("live") or {}
        return json.dumps({"score": (live.get("score") or {}).get("score"), "levels": progress.get("levels"),
                           "completion_tokens": live.get("completion_tokens")})


# -- controller VM -----------------------------------------------------------------------------------------------
def controller_alive(cloud, now=None, fresh=CLAIM_SECONDS):  # one blocked create must not make it look gone
    """A controller VM that reports and can create VMs: an unfit one relaunches nothing, so this side must."""
    beat = None if cloud.dry_run else cloud.read_json(CONTROLLER, "heartbeat.json")
    return bool(beat) and (now or time.time()) - beat.get("epoch", 0) < fresh and beat.get(
        "state") != "controller_unfit"


def controller_rate():
    return (2 * PRICES["cpu_d3"]["vcpu_hour"] + 8 * PRICES["cpu_d3"]["gib_hour"]
            + 50 * PRICES["disk_gib_month"]["network_ssd"] / 730)


def controller_up(root, config, cloud, run_id, hours=30.0, poll=20):
    """A small on-demand CPU VM that runs ``watch --controller``: it relaunches preempted runs from their bucket
    records with the service account alone (no key leaves SecretStash), deletes itself at its deadline or after an
    idle hour, and never touches the ledger (each run's reservation already covers its relaunches)."""
    seconds = int(hours * 3600)
    bound = bound_usd(seconds, controller_rate())
    runner = HERE / "nebius_runner.py"
    created = time.time()
    spec = {"run_id": run_id, "mode": "controller", "kind": "arena", "test": "controller", "timeout": seconds,
            "gpu": None, "bucket": config["bucket"], "runs_prefix": RUNS_PREFIX, "assets_prefix": ASSETS_PREFIX,
            "manifests_prefix": MANIFESTS_PREFIX, "launch_prefix": f"launch/{run_id}", "assets": [],
            "deadline_epoch": created + seconds, "finalize_allowance": 120, "sync_seconds": 60, "interval": 60,
            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(), "image_family": "ubuntu24.04-driverless"}
    entry = {"run_id": run_id, "state": "reserved", "kind": "controller", "platform": "nebius",
             "charged_bound_usd": bound, "max_seconds": seconds, "gpu": None, "nebius_platform": "cpu-d3",
             "preset": "2vcpu-8gb", "region": config["region"], "preemptible": False,
             "rate_usd_per_hour": controller_rate(),
             "rate_basis": "cpu-d3 2 vCPU / 8 GiB on-demand + 50 GiB network SSD",
             "authorization": "see nebius.cap_authorizations"}
    if cloud.dry_run:
        cloud.echo(f"DRY-RUN ledger: reserve ${bound:.2f} for {run_id} ({hours} h controller)")
    else:
        reserve(root, entry)
    record = {"state": "submission_intent", "run_id": run_id, "kind": "controller", "test": "controller",
              "platform": "nebius", "charged_bound_usd": bound, "created_epoch": created,
              "deadline_epoch": created + seconds, "preemptible": False, "rate": {"usd_per_hour": controller_rate()}}
    if not config.get("cli_version"):
        config["cli_version"] = cloud.cli_version()  # the controller creates VMs with this side's flags
    remote = {key: value for key, value in config.items() if key not in ("profile", "rclone_remote")}
    remote.update(profile=None, rclone_remote="arc", nebius_binary="/usr/local/bin/arc-nebius")
    cloud.rclone("copyto", str(runner), cloud.key(spec["launch_prefix"], "nebius_runner.py"))
    cloud.rclone("copyto", str(HERE / "nebius_validation.py"), cloud.key(spec["launch_prefix"], "nebius_validation.py"))
    cloud.put_text(json.dumps(remote, indent=1), spec["launch_prefix"], "config.json")
    name = slug(f"arc-{run_id}", 58)
    labels = {"arc-run": slug(run_id), "arc-deadline": str(int(created + seconds)), "arc-test": "controller"}
    user_data = render_user_data(spec, config)
    argv = create_command(config, name, labels, user_data, preemptible=False, platform="cpu-d3",
                          preset="2vcpu-8gb", disk=("network_ssd", 50), image_family="ubuntu24.04-driverless")
    created_vm = cloud.nebius(*argv, display=[*argv[:-1], "<user-data>"] if cloud.dry_run else None)
    if cloud.dry_run:
        return record
    record.update(state="submitted", instance_name=name, instance_id=instance_id_of(created_vm))
    write_record(root, record)
    wait_for_heartbeat(root, config, cloud, record)
    end = time.time() + config["heartbeat_seconds"]  # then its own report: can its CLI create this launcher's VMs?
    while time.time() < end:
        beat = cloud.read_json(CONTROLLER, "heartbeat.json") or {}
        if beat.get("epoch", 0) > created:
            record["controller"] = {key: beat.get(key) for key in ("state", "unfit", "cli_version")}
            write_record(root, record)
            print(json.dumps({"event": "controller_report", **record["controller"],
                              **({"next": "controller down: it cannot relaunch; `watch` here does"}
                                 if beat.get("state") == "controller_unfit" else {})}))
            break
        time.sleep(poll)
    return record


UNFIT_REPEATS = 3  # the same non-capacity create error this often: the controller cannot create, and says so


def controller_loop(config, cloud, interval=60, idle_seconds=3600, rounds=None):
    """On the controller VM: every ``interval`` advance each run that has a bucket record and is not over; leave
    after ``idle_seconds`` without one (the unit's end deletes the VM). A controller whose CLI lacks a create flag,
    or whose creates keep failing the same way, reports ``controller_unfit`` and only closes ended attempts (deletes
    their VMs, leaves the relaunch pending): `watch` on the laptop then relaunches."""
    idle_since, version, missing = None, cloud.cli_version(), cloud.missing_create_flags()
    unfit = ({"reason": "cli", "missing_flags": missing, "cli_version": version, "wanted": config.get("cli_version")}
             if missing or missing is None else None)

    def beat(active):
        state = "controller_unfit" if unfit else "ok"
        cloud.put_text(json.dumps({"epoch": round(time.time(), 1), "state": state, "unfit": unfit,
                                   "cli_version": version, "active": active, "idle_since_epoch": idle_since}),
                       CONTROLLER, "heartbeat.json")
        return state

    beat([])
    while rounds is None or rounds > 0:
        rounds = None if rounds is None else rounds - 1
        now, active = time.time(), []
        listing = cloud.rclone("lsf", cloud.key("launch") + "/", "--dirs-only", check=False, timeout=120)
        for name in (listing.stdout.split() if listing is not None else []):
            run_id = name.strip("/")
            record = cloud.read_json("launch", run_id, "record.json")
            if not record or record.get("state") in TERMINAL or not (
                    (record.get("policy") or {}).get("auto_relaunch") or record.get("state") == "relaunch_pending"):
                continue
            try:
                if unfit:
                    info, record = inspect(None, cloud, run_id, time.time())
                else:
                    info, record = advance(None, config, cloud, run_id, time.time())
                active.append({"run_id": run_id, "state": info["state"]})
                pending = record.get("pending") or {}
                if record.get("state") == "relaunch_pending" and pending.get("repeats", 0) >= UNFIT_REPEATS:
                    unfit = {"reason": "create", "error": pending.get("last_error"), "run_id": run_id,
                             "cli_version": version}
            except Exception as exc:  # noqa: BLE001 - one run's trouble must not stop the others
                active.append({"run_id": run_id, "error": f"{type(exc).__name__}: {str(exc)[-200:]}"})
        idle_since = None if active else (idle_since or now)
        state = beat(active)
        print(json.dumps({"event": "controller", "epoch": round(now, 1), "state": state, "unfit": unfit,
                          "active": active}), flush=True)
        if idle_since is not None and now - idle_since >= idle_seconds:
            return "idle"
        if rounds is None or rounds > 0:
            time.sleep(interval)
    return "rounds"


def reap(root, cloud, *, yes=False):
    rows, now = [], time.time()
    for item in cloud.instances():
        meta, state = item["metadata"], (item.get("status") or {}).get("state")
        labels = meta.get("labels") or {}
        run_label = labels["arc-run"]
        known = sorted(Path(root).glob(f"{os.environ.get('ARC_VALIDATION_RECORDS', RECORDS)}/nebius-*.json"))
        run_id = next((json.loads(p.read_text())["run_id"] for p in known
                       if slug(json.loads(p.read_text())["run_id"]) == run_label), None)
        final = cloud.read_json(RUNS_PREFIX, "validation", run_id, "nebius/final.json") if run_id else None
        reason = reap_reason(labels, state, final, now)
        if not reason and run_id and read_record(root, run_id).get("state") == "launch_failed_no_execution":
            reason = f"its launch failed (no execution); the create left it {str(state).lower()}"
        row = {"instance_id": meta["id"], "name": meta.get("name"), "run": run_id or run_label, "state": state,
               "deadline_in_seconds": int(labels.get("arc-deadline", "0") or 0) - int(now), "reap": reason}
        if reason and yes and run_id and read_record(root, run_id).get("attempts"):
            # A run with a relaunch policy: `inspect` deletes the VM and decides what follows (resume, fresh, settle).
            row["handled"] = inspect(root, cloud, run_id, now)[0]["state"]
        elif reason and yes:
            if run_id and not final and (state or "").upper() not in ("STOPPED", "ERROR"):
                cloud.put_text(json.dumps({"epoch": now, "reason": reason, "instance_state": state}) + "\n",
                               RUNS_PREFIX, "validation", run_id, "nebius/reap.json")
            cloud.delete(meta["id"])
            row["deleted"] = True
            if run_id:
                # A stopped VM without a final marker was stopped by the platform: preempted, not reaped.
                preempted = (state or "").upper() == "STOPPED" and not final
                record = read_record(root, run_id)
                if record.get("state") not in TERMINAL:
                    outcome = "preempted" if preempted else (final or {}).get("state", "reaped")
                    outcome = "preempted" if outcome == "interrupted" else outcome
                    entry = settle_entry(root, run_id, outcome, now - record["created_epoch"])
                    record.update(state=outcome, final=final, charged_bound_usd=entry["charged_bound_usd"])
                    write_record(root, record)
        rows.append(row)
    return rows


def preflight(root, config, cloud, run_id, seconds=900):
    """A CPU VM through the whole control path (no GPU, no assets): cents, and it must delete itself."""
    rate = (2 * PRICES["cpu_d3"]["vcpu_hour"] + 8 * PRICES["cpu_d3"]["gib_hour"]
            + 50 * PRICES["disk_gib_month"]["network_ssd"] / 730)
    bound = bound_usd(seconds, rate)
    runner = HERE / "nebius_runner.py"
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    created = time.time()
    spec = {"run_id": run_id, "mode": "preflight", "kind": "arena", "test": "preflight", "timeout": 0,
            "gpu": None, "bucket": config["bucket"], "runs_prefix": RUNS_PREFIX, "assets_prefix": ASSETS_PREFIX,
            "manifests_prefix": MANIFESTS_PREFIX, "launch_prefix": f"launch/{run_id}", "assets": [],
            "deadline_epoch": created + seconds, "finalize_allowance": 120, "sync_seconds": 30,
            "runner_sha256": runner_sha, "image_family": "ubuntu24.04-driverless"}
    entry = {"run_id": run_id, "state": "reserved", "kind": "preflight", "platform": "nebius",
             "charged_bound_usd": bound, "max_seconds": seconds, "gpu": None, "nebius_platform": "cpu-d3",
             "preset": "2vcpu-8gb", "region": config["region"], "preemptible": False, "rate_usd_per_hour": rate,
             "rate_basis": "cpu-d3 2 vCPU / 8 GiB on-demand + 50 GiB network SSD", "attempt": 1,
             "authorization": "see nebius.cap_authorizations"}
    if cloud.dry_run:
        cloud.echo(f"DRY-RUN ledger: reserve ${bound:.4f} for {run_id}")
    else:
        reserve(root, entry)
    record = {"state": "submission_intent", "run_id": run_id, "kind": "preflight", "test": "preflight",
              "platform": "nebius", "charged_bound_usd": bound, "created_epoch": created,
              "deadline_epoch": created + seconds, "preemptible": False}
    cloud.rclone("copyto", str(runner), cloud.key(spec["launch_prefix"], "nebius_runner.py"))
    name = slug(f"arc-{run_id}", 58)
    labels = {"arc-run": slug(run_id), "arc-deadline": str(int(created + seconds)), "arc-test": "preflight"}
    user_data = render_user_data(spec, config)
    argv = create_command(config, name, labels, user_data, preemptible=False, platform="cpu-d3",
                          preset="2vcpu-8gb", disk=("network_ssd", 50), image_family="ubuntu24.04-driverless")
    created_vm = cloud.nebius(*argv, display=[*argv[:-1], "<user-data>"] if cloud.dry_run else None)
    if cloud.dry_run:
        return record
    record.update(state="submitted", instance_name=name, instance_id=instance_id_of(created_vm))
    write_record(root, record)
    wait_for_heartbeat(root, config, cloud, record)
    end, final, gone = created + seconds, None, False
    while time.time() < end and not final:
        final = cloud.read_json(RUNS_PREFIX, "validation", run_id, "nebius/final.json")
        time.sleep(0 if final else 15)
    while final and time.time() < end and not gone:  # the unit's ExecStopPost must remove the VM by itself
        gone = (cloud.instance_state(record["instance_id"]) or "DELETING").upper() == "DELETING"
        time.sleep(0 if gone else 10)
    info, record = inspect(root, cloud, run_id)  # settles; deletes the VM if it is still there
    print(json.dumps({**info, "vm_deleted_itself": gone,
                      "checks": cloud.read_json(RUNS_PREFIX, "validation", run_id, "execution.json")}))
    if not final or not gone:
        raise RuntimeError("preflight failed: no final marker or the VM did not delete itself (it is deleted now)")
    return record


def cpujob(root, config, cloud, *, run_id, bundle, script, arguments, seconds, preset, inputs, authorization=None):
    """A CPU VM (cpu-d3, on-demand, no GPU, no assets) running ``script`` of ``bundle`` like a run's bootstrap, with
    its network open (pip) and ``inputs`` ({"runs": file, "assets": file}: lists of bucket keys under runs/validation
    and the assets prefix) copied to $ARC_INPUTS by the runner first. Outputs go to runs/validation/<run_id> as for
    any run. No wait: ``status <run_id>`` settles it (SFT arm, 16 Oct: corpus gathering next to the bucket)."""
    found = re.fullmatch(r"(\d+)vcpu-(\d+)gb", preset)
    if not found:
        raise ValueError(f"cpu-d3 preset like 16vcpu-64gb, not {preset!r}")
    vcpus, gib, disk_gib = int(found.group(1)), int(found.group(2)), 200
    rate = (vcpus * PRICES["cpu_d3"]["vcpu_hour"] + gib * PRICES["cpu_d3"]["gib_hour"]
            + disk_gib * PRICES["disk_gib_month"]["network_ssd"] / 730)
    setup = 900
    attempt_seconds = int(seconds) + setup + FINALIZE_SECONDS
    bound = bound_usd(attempt_seconds, rate)
    runner = HERE / "nebius_runner.py"
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    launch_prefix = f"launch/{run_id}"
    created = time.time()
    names = {part: f"inputs-{part}.txt" for part, path in (inputs or {}).items() if path}
    spec = {"run_id": run_id, "mode": "run", "kind": "cpujob", "test": "cpujob", "timeout": int(seconds),
            "script": script, "arguments": list(arguments), "stop_rule": None, "gpu": None, "bucket": config["bucket"],
            "runs_prefix": RUNS_PREFIX, "assets_prefix": ASSETS_PREFIX, "manifests_prefix": MANIFESTS_PREFIX,
            "launch_prefix": launch_prefix, "assets": [], "verify": config["verify"], "block_network": False,
            "finalize_allowance": FINALIZE_SECONDS, "sync_seconds": 60, "rate_per_second": rate / 3600,
            "runner_sha256": runner_sha, "image_family": "ubuntu24.04-driverless", "checkpoint": None,
            "exchange_in": [], "inputs": names, "attempt": 1, "launch_attempt": 1,
            "deadline_epoch": created + attempt_seconds}
    entry = {"run_id": run_id, "state": "reserved", "kind": "cpujob", "platform": "nebius", "charged_bound_usd": bound,
             "max_seconds": attempt_seconds, "gpu": None, "nebius_platform": "cpu-d3", "preset": preset,
             "region": config["region"], "preemptible": False, "rate_usd_per_hour": rate,
             "rate_basis": f"cpu-d3 {vcpus} vCPU / {gib} GiB on-demand + {disk_gib} GiB network SSD", "attempt": 1,
             "authorization": authorization or "see nebius.cap_authorizations"}
    record = {"state": "submission_intent", "run_id": run_id, "kind": "cpujob", "test": "cpujob", "platform": "nebius",
              "charged_bound_usd": bound, "created_epoch": created, "deadline_epoch": created + attempt_seconds,
              "preemptible": False, "bundle": str(bundle), "nebius_platform": "cpu-d3", "preset": preset,
              "outputs": f"s3://{config['bucket']}/{RUNS_PREFIX}/validation/{run_id}"}
    print(json.dumps({"run_id": run_id, "preset": preset, "rate_usd_per_hour": round(rate, 4),
                      "attempt_seconds": attempt_seconds, "charged_bound_usd": round(bound, 4), "inputs": names}))
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "bundle.tar.gz"
        spec["bundle_sha256"] = record["bundle_sha256"] = pack_bundle(Path(bundle).resolve(), archive)
        if cloud.dry_run:
            cloud.echo(f"DRY-RUN ledger: reserve ${bound:.4f} for {run_id}")
        else:
            reserve(root, entry)
            write_record(root, record)
        cloud.rclone("copyto", str(archive), cloud.key(launch_prefix, "bundle.tar.gz"))
        cloud.rclone("copyto", str(runner), cloud.key(launch_prefix, "nebius_runner.py"))
        for part, name in names.items():
            cloud.rclone("copyto", str(Path(inputs[part]).resolve()), cloud.key(launch_prefix, name))
    name = slug(f"arc-{run_id}", 58)
    labels = {"arc-run": slug(run_id), "arc-deadline": str(int(created + attempt_seconds)), "arc-test": "cpujob"}
    argv = create_command(config, name, labels, render_user_data(spec, config), preemptible=False, platform="cpu-d3",
                          preset=preset, disk=("network_ssd", disk_gib), image_family="ubuntu24.04-driverless")
    # a slow uplink (16 Oct: the CLI's 1 min request deadline ran out during a saturated upload)
    argv = [*argv[:-2], "--timeout", "10m", "--per-retry-timeout", "2m", *argv[-2:]]
    created_vm = cloud.nebius(*argv, display=[*argv[:-1], "<user-data>"] if cloud.dry_run else None, timeout=900)
    if cloud.dry_run:
        return record
    record.update(state="submitted", instance_name=name, instance_id=instance_id_of(created_vm))
    write_record(root, record)
    return record


SETUP = """One-time setup (run by the user; ids go to {config}, the key never touches this repo):
  nebius profile create arc --endpoint api.nebius.cloud --federation-endpoint auth.nebius.com \\
      --parent-id <project_id>                                   # browser login
  nebius vpc subnet list --parent-id <project_id>                # -> subnet_id
  nebius iam service-account create --parent-id <project_id> --name arc-runner         # -> service_account_id
  nebius iam group create --parent-id <project_id> --name arc-runners                  # -> group id
  nebius iam access-permit create --parent-id <group_id> --resource-id <project_id> --role editor
  nebius iam group-membership create --parent-id <group_id> --member-id <service_account_id>
  nebius storage bucket create --parent-id <project_id> --name <bucket> \
      --default-storage-class enhanced_throughput --force-storage-class=true   # eu-south1/uk-south2: standard quota is 0
  nebius iam v2 access-key create --account-service-account-id <service_account_id> --description arc-bucket
  nebius iam v2 access-key get --id <access_key_id>   # status.aws_access_key_id; the secret: see the docs page
  rclone config create {remote} s3 provider=Other env_auth=false access_key_id=<id> secret_access_key=<secret> \\
      endpoint={endpoint} region={region}
  nebius mysterybox secret create --parent-id <project_id> --name arc-bucket-key --secret-version-payload \\
      '[{{"key":"AWS_ACCESS_KEY_ID","string_value":"<id>"}},{{"key":"AWS_SECRET_ACCESS_KEY","string_value":"<secret>"}}]'
  nebius iam access-permit create --parent-id <group_id> --resource-id <secret_id> --role mysterybox.payload-viewer
  nebius billing pricing-policy create --parent-id <project_id> --name arc-rtx6000 \\
      --compute-instance-spec-v1-platform {platform} --pricing-max-price-v1-max-price <max USD per GPU hour>
  modal secret create nebius-s3 AWS_ACCESS_KEY_ID=<id> AWS_SECRET_ACCESS_KEY=<secret>   # for the asset mirror
Config template ({config}):
{template}
"""


def image_bundle(dest, entry, image, seconds, assets, arguments=(), env=None):
    """A launchable bundle for the ``image`` backend: the entry script, configs/comparison.json naming the image and
    its assets, and the MANIFEST verify_bundle checks. Launch it as any run (ARC_VALIDATION_KIND=arena,
    ARC_VALIDATION_TEST=image, ARC_VALIDATION_BUNDLE=dest): the gate, ledger, watcher and stop work as for arena runs."""
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    (dest / "configs").mkdir(parents=True)
    shutil.copyfile(entry, dest / entry.name)
    study = {"backend": "image", "image": {"ref": image, "entry": entry.name, "arguments": list(arguments),
                                           "timeout_seconds": int(seconds), "assets": list(assets), "env": env or {}}}
    plan_run("arena", "image", study, "image-bundle", dest)  # the launcher's own checks, now
    (dest / "configs/comparison.json").write_text(json.dumps(study, indent=1) + "\n")
    files = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(dest.rglob("*")) if p.is_file()}
    (dest / "MANIFEST.json").write_text(json.dumps({"files": files, "dependencies_prepared": True,
                                                    "source": "nebius_validation image-bundle"}, indent=1) + "\n")
    return {"bundle": str(dest), "image": image, "entry": entry.name, "assets": list(assets), "files": len(files)}


def bake_image(root, cloud, run_id, *, yes=False, poll=20, limit=7200):
    """A finished bake (final state "baked", VM powered off) -> a compute image of its boot disk (family
    arc-bake-<run>), then the VM and its disk deleted. Records the image id, its size and the seconds the image took
    in the run's record. Without --yes: what it would do."""
    record = read_record(root, run_id)
    final = cloud.read_json(RUNS_PREFIX, "validation", record["run_id"], "nebius/final.json") or {}
    instance_id = record.get("instance_id")
    got = cloud.nebius("compute", "instance", "get", "--id", instance_id, check=False) if instance_id else None
    state = ((got or {}).get("status") or {}).get("state")
    disk = (((got or {}).get("spec") or {}).get("boot_disk") or {}).get("existing_disk", {}).get("id") \
        or (((got or {}).get("status") or {}).get("boot_disk") or {}).get("id")
    if not disk and got:
        for item in (cloud.nebius("compute", "disk", "list", "--parent-id", cloud.config["project_id"]) or {}).get(
                "items", []):
            if item["metadata"].get("name") == f"{got['metadata']['name']}-boot":
                disk = item["metadata"]["id"]
    plan = {"run_id": record["run_id"], "final_state": final.get("state"), "instance": instance_id,
            "instance_state": state, "disk": disk, "image_family": slug(f"arc-bake-{record['run_id']}", 63)}
    if final.get("state") != "baked" or (state or "").upper() != "STOPPED" or not disk:
        raise RuntimeError(f"not a powered-off finished bake: {plan}")
    if not yes:
        return {**plan, "next": "bake-image --yes creates the image, then deletes the VM and its disk"}
    started = time.time()
    # a guest power-off leaves spec.stopped false, and the image API refuses a disk attached to a "not-stopped"
    # instance (first bake, 21 Oct): stop it through the API too
    cloud.nebius("compute", "instance", "stop", "--id", instance_id)
    made = cloud.nebius("compute", "image", "create", "--parent-id", cloud.config["project_id"],
                        "--name", plan["image_family"], "--image-family", plan["image_family"], "--source-disk-id", disk,
                        timeout=limit)
    image = ((made or {}).get("metadata") or {}).get("id") or (made or {}).get("resource_id")
    info = None
    while image and time.time() - started < limit:
        info = cloud.nebius("compute", "image", "get", "--id", image, check=False)
        if ((info or {}).get("status") or {}).get("state", "").upper() == "READY":
            break
        time.sleep(poll)
    status = (info or {}).get("status") or {}
    result = {**plan, "image_id": image, "image_state": status.get("state"),
              "image_size_bytes": status.get("storage_size_bytes") or status.get("size_bytes"),
              "image_seconds": round(time.time() - started, 1)}
    if (status.get("state") or "").upper() != "READY":
        raise RuntimeError(f"image not ready after {limit} s: {result}")
    cloud.delete(instance_id)
    record["bake_image"] = result
    write_record(root, record)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("launch", "status", "pull", "stop", "watch", "list", "reap", "relaunch", "preflight", "set-cap",
                "mirror-plan", "setup", "controller", "cpujob", "bake-image", "image-bundle")
    if not argv or argv[0] not in commands:
        argv.insert(0, "launch")  # the Modal launcher's interface: environment + --run-id
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=commands)
    parser.add_argument("target", nargs="?")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="print every cloud call instead of making it")
    parser.add_argument("--preemptible", action="store_true",
                        help="a spot VM with automatic relaunch and checkpoint resume (default: a fixed instance)")
    parser.add_argument("--on-demand", action="store_true", help="the default; kept for older command lines")
    parser.add_argument("--no-wait", action="store_true", help="skip the heartbeat wait after the create")
    parser.add_argument("--reason", help="stop: why; launch: why the mechanism needs more than 2 h of play")
    parser.add_argument("--authorization")
    parser.add_argument("--approval", help="the user's words approving more than 4 h of play")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dest")
    parser.add_argument("--no-relaunch", action="store_true", help="launch: one attempt only; watch: observe only")
    parser.add_argument("--fresh", action="store_true", help="relaunch: from zero under <run>-a<N>, not from a checkpoint")
    parser.add_argument("--controller", action="store_true", help="watch: the controller VM's loop over every run")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--checkpoint-seconds", type=int,
                        help="launch: this run's checkpoint interval (preemptible default 600; fixed: off without it)")
    parser.add_argument("--hours", type=float, default=30.0, help="controller up: its lifetime (reserved)")
    parser.add_argument("--interval", type=int)
    parser.add_argument("--script", help="cpujob: the bundle-relative script the runner starts")
    parser.add_argument("--job-args", default="", help="cpujob: the script's extra arguments (one shell-split string)")
    parser.add_argument("--seconds", type=int, default=3600, help="cpujob: the script's timeout")
    parser.add_argument("--preset", default="16vcpu-64gb", help="cpujob: the cpu-d3 preset")
    parser.add_argument("--inputs-runs", help="cpujob: file of bucket keys under runs/validation to copy in")
    parser.add_argument("--inputs-assets", help="cpujob: file of bucket keys under the assets prefix to copy in")
    parser.add_argument("--image", help="image-bundle: the docker image, digest-pinned (@sha256:) or a version tag")
    parser.add_argument("--assets", default="", help="image-bundle: comma-separated bucket asset prefixes (-> /w/<prefix>)")
    parser.add_argument("--env", action="append", default=[], help="image-bundle: KEY=VALUE for the container")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("ARC_VALIDATION_ROOT") or HERE.parent).resolve()
    if root.parent.name == "worktrees" and root.parent.parent.name == ".claude":  # records live in main only
        root = root.parents[2]
    run_id = args.run_id or args.target

    if args.command == "image-bundle":  # --dest DIR --script entry.sh --image REF --seconds S [--assets a,b] ...
        return print(json.dumps(image_bundle(Path(args.dest), Path(args.script), args.image, args.seconds,
                                             [a for a in args.assets.split(",") if a], shlex.split(args.job_args),
                                             dict(e.split("=", 1) for e in args.env))))
    if args.command == "set-cap":
        set_cap(root, float(args.target), args.authorization or "")
        return print(json.dumps({"nebius_cap_usd": float(args.target)}))
    if args.command == "setup":
        template = {**{key: f"<{key}>" for key in REQUIRED}, "region": "eu-south1", "profile": "arc",
                    "pricing_policy_id": "<pricing_policy_id>", "spot_max_price_usd_per_gpu_hour": 1.2,
                    "rclone_remote": DEFAULTS["rclone_remote"]}
        return print(SETUP.format(config=os.environ.get("ARC_NEBIUS_CONFIG", "~/.nebius/arc.json"),
                                  remote=DEFAULTS["rclone_remote"], region="<region>", platform="<gpu-rtx6000[-a]>",
                                  endpoint="https://storage.<region>.nebius.cloud",
                                  template=json.dumps(template, indent=1)))
    config = load_config()
    if args.checkpoint_seconds:
        config["checkpoint_seconds"] = max(60, args.checkpoint_seconds)
    cloud = Cloud(config, dry_run=args.dry_run)
    kind = os.environ.get("ARC_VALIDATION_KIND", "arena")
    test = os.environ.get("ARC_VALIDATION_TEST", "smoke")
    bundle = Path(os.environ.get("ARC_VALIDATION_BUNDLE", "dist/arena-25g-900s-v2")).resolve()

    if args.command == "watch" and args.controller:
        return print(controller_loop(config, cloud, interval=args.interval or 60))
    if args.command == "controller":
        if args.target == "up":
            name = args.run_id or f"controller-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
            return print(json.dumps(controller_up(root, config, cloud, name, args.hours)))
        beat = cloud.read_json(CONTROLLER, "heartbeat.json")
        rows = [i for i in cloud.instances() if (i["metadata"].get("labels") or {}).get("arc-test") == "controller"]
        if args.target == "down":
            for item in rows:
                cloud.delete(item["metadata"]["id"])
            return print(json.dumps({"deleted": [i["metadata"]["id"] for i in rows],
                                     "next": "status <controller run id> settles its ledger entry"}))
        return print(json.dumps({"alive": controller_alive(cloud), "heartbeat": beat,
                                 "instances": [{"id": i["metadata"]["id"], "name": i["metadata"].get("name"),
                                                "state": (i.get("status") or {}).get("state")} for i in rows]}, indent=1))
    if args.command == "bake-image":
        return print(json.dumps(bake_image(root, cloud, run_id, yes=args.yes), indent=1))
    if args.command == "mirror-plan":
        study = json.loads((bundle / "configs/comparison.json").read_text())
        external = study.get("backend") == "dfr-exact"  # its inputs live in the arc-dfr-exact volume
        assets = [str(a).strip("/") for a in study["dfr_exact"]["assets"]] if external else required_assets(study)
        listing = cloud.rclone("lsf", cloud.key(MANIFESTS_PREFIX) + "/", check=False, timeout=120)
        have = set(listing.stdout.split()) if listing is not None else set()
        missing = [prefix for prefix in assets if f"{asset_slug(prefix)}.json" not in have]
        print(json.dumps({"bundle": str(bundle), "required": assets, "missing": missing}, indent=1))
        if missing:
            print("uv run --extra modal modal run --detach scripts/nebius_mirror.py --bucket "
                  f"{config['bucket']} --endpoint {config['endpoint']} --region {config['region']} "
                  f"--prefixes {','.join(missing)}" + (" --source dfr-exact" if external else ""))
        return None
    if args.command == "cpujob":
        return print(json.dumps(cpujob(root, config, cloud, run_id=run_id, bundle=bundle, script=args.script,
                                       arguments=shlex.split(args.job_args), seconds=args.seconds, preset=args.preset,
                                       inputs={"runs": args.inputs_runs, "assets": args.inputs_assets},
                                       authorization=args.authorization), indent=1))
    if args.command in ("list", "reap"):
        rows = reap(root, cloud, yes=args.command == "reap" and args.yes)
        return print(json.dumps(rows, indent=1))
    if args.command == "watch" and not run_id:  # every run with a receipt here that is not over
        while True:
            active = [json.loads(p.read_text()) for p in sorted(root.glob(
                f"{os.environ.get('ARC_VALIDATION_RECORDS', RECORDS)}/nebius-*.json"))]
            active = [r["run_id"] for r in active if "attempts" in r and r.get("state") not in TERMINAL
                      and r.get("state") not in ("launch_failed_no_execution", "failed_no_heartbeat")]
            for name in active:
                info, _ = (inspect(root, cloud, name) if args.no_relaunch else advance(root, config, cloud, name))
                print(time.strftime("%H:%M:%S"), json.dumps({k: info.get(k) for k in ("run_id", "state", "phase", "attempt")}),
                      flush=True)
            if not active:
                return None
            time.sleep(args.interval or 60)
    if not run_id:
        raise ValueError("--run-id is required")
    if args.command == "launch":
        return launch(root, config, cloud, run_id=run_id, kind=kind, test=test, bundle=bundle,
                      preemptible=args.preemptible and not args.on_demand, wait=not args.no_wait,
                      checkpoint=True if args.checkpoint_seconds else None, authorization=args.authorization,
                      max_attempts=args.max_attempts, approval=args.approval, relaunch=not args.no_relaunch,
                      reason=args.reason)
    if args.command == "preflight":
        return preflight(root, config, cloud, run_id)
    if args.command == "stop":
        if not args.reason:
            raise SystemExit('stop needs --reason "<why>" (it is recorded with the run); nothing was stopped')
        current = (load_record(root, cloud, run_id).get("attempts") or [{}])[-1].get("run_id", run_id)
        payload = json.dumps({"reason": args.reason, "epoch": round(time.time(), 1)}) + "\n"
        cloud.put_text(payload, RUNS_PREFIX, "validation", current, "arena/live/stop-request")
        return print(json.dumps({"path": f"validation/{current}/arena/live/stop-request",
                                 "requested": not args.dry_run}))
    if args.command == "pull":
        dest = Path(args.dest or root / "runs/nebius/validation" / run_id)
        source = cloud.key(RUNS_PREFIX, "validation", run_id, *(["arena/live"] if args.live else []))
        cloud.rclone("copy", source, str(dest / "arena/live" if args.live else dest), "--transfers", "16",
                     "--exclude", "arena/working/trials/*/checkpoint/**")
        return print(dest)
    if args.command == "relaunch":  # by hand: the pending relaunch now (or a fresh restart of a preempted run)
        info, record = advance(root, config, cloud, run_id, fresh=args.fresh)
        return print(json.dumps({**info, "attempts": [{k: a.get(k) for k in ("n", "run_id", "kind", "state")}
                                                      for a in record.get("attempts") or []]}))
    while True:  # status, watch
        step = inspect if args.command == "status" or args.no_relaunch else (
            lambda *a: advance(a[0], config, *a[1:]))
        info, record = step(root, cloud, run_id)
        attempts = record.get("attempts") or []
        seconds = sum(a.get("observed_seconds", 0) for a in attempts[:-1]) + (
            attempts[-1].get("observed_seconds", time.time() - record["created_epoch"]) if attempts
            else 0 if "attempts" in record else time.time() - record["created_epoch"])
        current = attempts[-1]["run_id"] if attempts else run_id
        saved = cloud.read_json(RUNS_PREFIX, "validation", current, "nebius/checkpoint.json") if (
            record.get("policy") or {}).get("checkpointing") else None
        print(time.strftime("%H:%M:%S"), json.dumps({
            **info, "attempts": [f"{a['n']}:{a['kind']}:{a.get('state', 'running')}" for a in attempts],
            "checkpoint": (saved or {}).get("trials"),
            "usd_so_far": round(seconds * record.get("rate", {}).get("usd_per_hour", 0) / 3600, 2)
            if record.get("state") not in TERMINAL else record.get("charged_bound_usd")}))
        text = live_summary(cloud, current)
        if text:
            print(text, flush=True)
        if args.command == "status" or record.get("state") in TERMINAL:
            return
        time.sleep(args.interval or 60)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as failure:  # the launcher's own refusals: a line, no trace
        if os.environ.get("ARC_NEBIUS_TRACEBACK"):
            raise
        raise SystemExit(f"error: {failure}") from None
