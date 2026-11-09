"""Isolation self-test (run by the loop at every start, before any session; a failure stops the run).

A probe runs inside the real sandbox of each role, built by the same code the loop uses: a player session (slot user,
its namespace), the proposer, a broker (a slot's game user with one game folder) and an unprivileged serving stack
(rsiserve). From inside it tries to read the loop's state, checkpoint, events, held-out games and results, other
sessions' folders and the run output; to write anywhere outside its own folders; to see files the root left in the
host's /tmp, /var/tmp and /dev/shm; to reach the model server's port directly and the internet. Players and the
proposer must reach the metering proxy (and only through it).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import json, os, socket, sys
spec = json.loads(sys.argv[1])
out = {"uid": os.getuid(), "fail": [], "ok": []}
def check(name, passed, detail=""):
    (out["ok"] if passed else out["fail"]).append(name + (": " + detail if detail and not passed else ""))
for path in spec["no_read"]:
    try:
        if os.path.isdir(path):
            os.listdir(path)
            seen = [n for n in os.listdir(path)]
            check("hidden " + path, not seen, "listed %d entries" % len(seen))
        else:
            open(path, "rb").read(1)
            check("hidden " + path, False, "readable")
    except OSError:
        check("hidden " + path, True)
for path in spec["no_write"]:
    target = os.path.join(path, ".rsi-probe-%d" % os.getpid())
    try:
        with open(target, "w") as handle:
            handle.write("x")
        os.unlink(target)
        check("read-only " + path, False, "writable")
    except OSError:
        check("read-only " + path, True)
for path in spec["absent"]:
    check("private " + path, not os.path.exists(path), "host file visible")
for path in spec["must_write"]:
    target = os.path.join(path, ".rsi-probe-%d" % os.getpid())
    try:
        with open(target, "w") as handle:
            handle.write("x")
        os.unlink(target)
        check("writable " + path, True)
    except OSError as exc:
        check("writable " + path, False, str(exc))
for host, port in spec["refused"]:
    s = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    s.settimeout(3)
    try:
        s.connect((host, port))
        check("no route %s:%d" % (host, port), False, "connected")
    except OSError:
        check("no route %s:%d" % (host, port), True)
    finally:
        s.close()
for host, port in spec["reachable"]:
    s = socket.socket()
    s.settimeout(5)
    try:
        s.connect((host, port))
        s.sendall(b"GET /health HTTP/1.0\r\nHost: x\r\nAuthorization: Bearer rsi-isolation-probe\r\n\r\n")
        reply = s.recv(64)
        check("proxy %s:%d" % (host, port), reply.startswith(b"HTTP/"), repr(reply[:40]))
    except OSError as exc:
        check("proxy %s:%d" % (host, port), False, str(exc))
    finally:
        s.close()
for path in spec["readable"]:
    check("visible " + path, os.path.exists(path), "missing")
if spec.get("only_loopback"):
    devices = [line.split(":")[0].strip() for line in open("/proc/self/net/dev").read().splitlines()[2:]]
    check("own network namespace (loopback only)", devices == ["lo"], str(devices))
    routes = [line.split() for line in open("/proc/self/net/route").read().splitlines()[1:] if line.strip()]
    check("no IPv4 route", not routes, str(routes[:3]))
    try:
        v6 = {line.split()[-1] for line in open("/proc/self/net/ipv6_route").read().splitlines() if line.strip()}
    except OSError:
        v6 = set()
    check("no IPv6 route off loopback", v6 <= {"lo"}, str(sorted(v6)))
print(json.dumps(out))
"""


def marker_paths():
    return [f"/tmp/rsi-isolation-{os.getpid()}", f"/var/tmp/rsi-isolation-{os.getpid()}",
            f"/dev/shm/rsi-isolation-{os.getpid()}"]


def spec_for(loop, role):
    rt = loop.runtime
    secret = [loop.ckpt / "state.json", loop.ckpt / "events.jsonl", loop.ckpt, loop.output, rt.root / "root",
              rt.root / "games", rt.root / "slots", rt.root / "prop", rt.root / "serve", rt.root / "sock",
              rt.bundle / "rsi/games/heldout_a", rt.bundle / "rsi/games/heldout_b",
              rt.bundle / "configs/rsi/heldout_a_points.json", "/runs", "/tmp/arena", "/opt/arc", "/root"]
    if loop.games_dir:
        secret.append(loop.games_dir)
    no_read = [str(p) for p in secret]
    no_write = ["/", "/etc", "/usr", "/opt", "/var/lib", "/var/crash", "/run/lock", "/run", "/home",
                str(Path(__file__).parents[2])]
    spec = {"no_read": no_read, "no_write": no_write, "absent": marker_paths(), "must_write": ["/tmp", "/dev/shm"],
            "refused": [["<ip>", 443], ["<ip>", 53]], "reachable": [], "readable": []}
    if role in ("player", "proposer", "game"):
        spec["only_loopback"] = True  # no egress is possible from these: the frontier audit's proof for them
    if role == "player":
        spec["no_read"] += ["/workspace/leak_state", "/workspace/leak_games/.", "/home/session/leak_root/."]
    if role in ("player", "proposer"):
        spec["refused"].append(["127.0.0.1", loop.server_port])
        spec["reachable"].append(["127.0.0.1", loop.proxy.port])
        spec["must_write"].append("/workspace")
    if role == "game":
        spec["refused"].append(["127.0.0.1", loop.server_port])
        spec["readable"].append("/game/environment_files")
    if role == "server":
        # the server's own working folder and venv are bound into its private /tmp (often both under /tmp/arena):
        # /tmp/arena itself is then listable; everything else the host keeps there must stay unseen
        own = [str(loop.stacks.server_dir), sys.prefix]
        if "/tmp/arena" in spec["no_read"] and any(o.startswith("/tmp/arena/") for o in own):
            spec["no_read"].remove("/tmp/arena")
            spec["no_read"] += [str(p) for p in sorted(Path("/tmp/arena").iterdir())
                                if not any(o == str(p) or o.startswith(str(p) + "/") for o in own)]
        spec["readable"] += [p for p in ("/dev/nvidia0", "/dev/nvidiactl", "/kaggle/input", "/serve/src/arena")
                             if Path(p).exists() or p.startswith("/serve")]
    return spec


async def run_probe(argv, timeout=120):
    process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        return {"fail": ["probe timed out"], "ok": []}
    lines = [line for line in out.decode(errors="replace").splitlines() if line.startswith("{")]
    if not lines:
        return {"fail": [f"probe did not run: exit {process.returncode}: {err.decode(errors='replace')[-600:]}"],
                "ok": []}
    return json.loads(lines[-1])


def link_attack_tree(loop):
    """A candidate tree whose files link to the loop's secrets (and one link inside it), for the root-side checks and
    the player probe: the root must refuse it, and a slot made from it must hold links, never their targets."""
    import shutil

    tree = loop.root / "root/isolation/tree"
    shutil.rmtree(tree.parent, ignore_errors=True)
    gen0 = loop.candidate_dir("c0000")
    (tree / "player/workspace").mkdir(parents=True)
    (tree / "player/home").mkdir(parents=True)
    (tree / "serving").mkdir(parents=True)
    for rel in ("player/run.json", "serving/server.json"):
        shutil.copy2(gen0 / rel, tree / rel)
    os.symlink(str(loop.ckpt / "state.json"), tree / "player/workspace/leak_state")
    os.symlink(str(loop.root / "games/heldout_a"), tree / "player/workspace/leak_games")
    os.symlink(str(loop.root / "root"), tree / "player/home/leak_root")
    os.symlink(str(loop.ckpt / "state.json"), tree / "player/game-prompt.txt")
    os.symlink("../run.json", tree / "player/workspace/inside")  # allowed: it stays in the tree
    return tree


def root_checks(loop):
    """What the root does with files the proposer or a game controls never reads through a link."""
    import shutil

    from . import candidates as cand
    from .attempt import Attempt
    ok, fail = [], []

    def check(name, passed):
        (ok if passed else fail).append(name)

    tree = link_attack_tree(loop)
    links = cand.escaping_links(tree)
    check("check() refuses links out of a candidate", bool(cand.check(tree)) and len(links) == 4)
    check("an inside link is allowed", "player/workspace/inside" not in links)
    copy = tree.parent / "archived"
    cand.copy_untrusted(tree.parent, tree.name, copy, link_root_readonly=True)
    # the proposer's workspace: a submitted name that is a link, a linked folder on the way, planted files
    ws = tree.parent / "ws"
    (ws / "work/real").mkdir(parents=True)
    secret = tree.parent / "secret"
    (secret / "sub").mkdir(parents=True)
    (secret / "sub/held.txt").write_text("held-out")
    os.symlink(str(secret), ws / "work/top")
    os.symlink(str(secret / "sub"), ws / "work/real/inner")
    for name, rel in (("submit refuses a name that is a link", "work/top"),):
        try:
            cand.copy_untrusted(ws, rel, tree.parent / "c1")
            check(name, False)
        except (OSError, ValueError):
            check(name, not (tree.parent / "c1/sub/held.txt").exists())
    cand.copy_untrusted(ws, "work/real", tree.parent / "c2")
    check("a linked folder inside is copied as a link", (tree.parent / "c2/inner").is_symlink())
    os.symlink(str(secret / "sub"), ws / "work2")
    try:
        cand.open_dir(ws, "work2/held")
        check("a linked folder on the way is refused", False)
    except (OSError, ValueError):
        check("a linked folder on the way is refused", True)
    target = secret / "sub/held.txt"
    os.symlink(str(target), ws / "events.jsonl")
    cand.write_untrusted(ws, "events.jsonl", b"row\n", append=True)
    check("an append never writes through a planted link", target.read_text() == "held-out"
          and not (ws / "events.jsonl").is_symlink())
    os.symlink(str(target), ws / "AGENTS.md")
    cand.write_untrusted(ws, "AGENTS.md", b"guidance")
    check("a write never writes through a planted link", target.read_text() == "held-out")
    staging = tree.parent / "staging"
    staging.mkdir()
    shutil.rmtree(ws / "work")
    os.symlink(str(secret), ws / "work")
    try:
        cand.move_into(staging, ws, "work", "fork")
        check("a fork never lands through a linked work folder", False)
    except (OSError, ValueError):
        check("a fork never lands through a linked work folder", not (secret / "fork").exists())
    check("archive copy keeps links as links", (copy / "player/workspace/leak_state").is_symlink())
    try:
        cand.read_inside(tree, "player/game-prompt.txt")
        check("prompt read refuses a link out", False)
    except (OSError, ValueError):
        check("prompt read refuses a link out", True)
    try:
        cand.copy_regular(tree / "player/game-prompt.txt", tree.parent / "copied")
        check("record copy refuses a link", False)
    except (OSError, ValueError):
        check("record copy refuses a link", True)
    point = loop.points["practice"][sorted(loop.points["practice"])[0]]
    job = {"id": "isolation-links", "slot": 1, "candidate": "c0000", "kind": "practice", "game": point["game"]}
    attempt = Attempt(loop.runtime, job, point, tree, wall=60, meter="loop", games_set="practice")
    attempt._fresh()
    slot = attempt.slot
    check("slot holds the links, not their targets", (slot / "workspace/leak_state").is_symlink()
          and (slot / "workspace/leak_games").is_symlink() and (slot / "home/leak_root").is_symlink())
    shutil.rmtree(slot, ignore_errors=True)
    shutil.rmtree(attempt.sock, ignore_errors=True)
    return {"ok": ok, "fail": fail}


async def self_test(loop):
    """Every role's probe; returns {role: result}. Raises when any check fails."""
    from . import sandbox
    from .attempt import RUNTIME, Attempt

    python = str(RUNTIME / "kernel/bin/python")
    markers = marker_paths()
    for path in markers:
        Path(path).write_text("root-only marker")
    results = {}
    loop.proxy.register("rsi-isolation-probe", "isolation-probe", "loop")
    try:
        practice = sorted(loop.points.get("practice", {}))
        point = loop.points["practice"][practice[0]]
        job = {"id": "isolation-probe", "slot": 0, "candidate": "c0000", "kind": "practice", "game": point["game"]}
        attempt = Attempt(loop.runtime, job, point, loop.candidate_dir("c0000"), wall=60, meter="loop",
                          games_set="practice")
        results["root"] = root_checks(loop)
        attempt = Attempt(loop.runtime, job, point, link_attack_tree(loop), wall=60, meter="loop",
                          games_set="practice")  # the player probe runs in a slot made from the linking tree
        attempt._fresh()
        probe = lambda role: [python, "-I", "-c", PROBE, json.dumps(spec_for(loop, role))]  # noqa: E731
        results["player"] = await run_probe(attempt._session_argv("probe", program=probe("player")))
        attempt.tree = loop.candidate_dir("c0000")
        results["game"] = await run_probe(attempt.broker_argv(program=probe("game")))
        results["proposer"] = await run_probe(loop.proposer.argv(loop.state["meta"]["live"], "probe", "probe",
                                                                 program=probe("proposer")))
        if loop.stacks.runtime is not None:
            results["server"] = await run_probe(loop.stacks.unprivileged_argv(
                loop.candidate_dir("c0000") / "serving", probe("server")))
        for user in (sandbox.slot_user(0), sandbox.game_user(0), sandbox.PROPOSER, sandbox.SERVE):
            subprocess.run(["pkill", "-KILL", "-u", user], check=False)
        import shutil
        shutil.rmtree(attempt.slot, ignore_errors=True)
        shutil.rmtree(attempt.sock, ignore_errors=True)
    finally:
        loop.proxy.forget("rsi-isolation-probe")
        for path in markers:
            Path(path).unlink(missing_ok=True)
    failed = {role: r["fail"] for role, r in results.items() if r.get("fail")}
    loop.event_log({"event": "rsi_isolation", "ok": not failed,
                    "checks": {role: len(r.get("ok", [])) for role, r in results.items()}, "failed": failed})
    if failed:
        raise RuntimeError(f"isolation self-test failed: {json.dumps(failed)[:3000]}")
    return results
