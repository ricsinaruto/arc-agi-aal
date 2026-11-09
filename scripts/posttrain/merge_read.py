"""Merge read verdicts (pre-registered 19 Oct; ``merge_lora.verdicts``): per arm, sb26 L2 after the agent's own L1
(lead point); L2 tokens per clear, path works (zero / merged >= 2.0), chain OK (zero / shipped CI contains 1).
usage: merge_read.py shipped=RUN zero=RUN merged=RUN  (Nebius run ids; reads level-bench.json from the bucket)"""
import json
import subprocess
import sys

from merge_lora import verdicts

B = "nebius-arc:arc-runs-bucket/runs/validation/"


def rows(run):
    out = subprocess.run(["rclone", "cat", B + run + "/arena/trials/level-bench/level-bench.json"], capture_output=True)
    return json.loads(out.stdout)["attempts"]


if __name__ == "__main__":
    print(json.dumps(verdicts({k: rows(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}), indent=1))
