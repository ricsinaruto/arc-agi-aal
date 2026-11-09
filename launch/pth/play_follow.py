"""play_follow.py PLAY_RUN TRAINER_RUN [START_VERSION]: repoint a running OPD play to a relaunched trainer's weights
(bucket validation/<play>/stream/follow.json; the runner copies <trainer>/deltas into the play's fixed folder, the
follower resets its version floor). Only for a trainer that moved nothing past its starting weights; else the play
halts following and needs a restart (weights-follow.jsonl 'halted'); refused here unless --force."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
import nebius_validation as nv  # noqa: E402

force = "--force" in sys.argv
args = [a for a in sys.argv[1:] if a != "--force"]
play, trainer = args[0], args[1]
start = int(args[2]) if len(args) > 2 else 0
cloud = nv.Cloud(nv.load_config())
current = (nv.load_record(Path("."), cloud, play).get("attempts") or [{}])[-1].get("run_id", play)
# c5/c6 (23 Oct): repointing a play that had applied a training update halted its follower silently; refuse instead
r = cloud.rclone("cat", cloud.key(nv.RUNS_PREFIX, "validation", current,
                                  "arena/working/trials/duck-transfer/weights-follow.jsonl"), check=False, timeout=120)
rows = [json.loads(x) for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else None
if rows is None and not force:
    sys.exit(f"REFUSED: cannot read {current}'s weights-follow.jsonl (rc {r.returncode}); --force to repoint blind")
done = [x for x in rows or [] if x.get("kind") in ("applied", "halted")]
if done and not force:
    sys.exit(f"REFUSED: {current} has already {done[-1]['kind']} (version {done[-1].get('version')}); its follower halts on "
             "a repoint after a training update: restart the play instead")
payload = json.dumps({"in": f"{trainer}/deltas", "generation": int(time.time()), "start_version": start}) + "\n"
cloud.put_text(payload, nv.RUNS_PREFIX, "validation", current, "stream/follow.json")
print(json.dumps({"path": f"validation/{current}/stream/follow.json", "follow": json.loads(payload)}))
