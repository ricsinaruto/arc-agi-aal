"""febench GPU timeout from measured phase times (checklist 3; mfu2-aot2 lost its last arm to a hand-set 2,400 s).
B300, bundle v6m, measured 25 Oct: setup (image, venv 40 s, prompts) ~180 s; boot to ready R-0 341 s, J1/J3 247-253,
J2 318, J0 460, W2 with a 120 GiB KV pool + probe ~600 (sleepkv5: 16 min for R-0 + W2); a decode-checked arm adds
~90 s (12 prompts x 2-3 modes, stop). Unknown kinds take the slowest boot. Need = 1.3 x sum + 300 s.

    python sizing.py R-0,J1,W3 --mode tput [--warm N --measure N] [--cpu]   -> prints the seconds (0 with --cpu)"""
import argparse
import re

SETUP_S = 180
BOOT_S = {"R": 360, "GR": 360, "J1": 260, "J3": 260, "J2": 330, "J0": 470}
SLOW_BOOT_S = 480
DECODE_S = 90
SLEEP_PROBE_S = {"W0": 240, "W1": 300, "W2": 300, "W3": 240}  # sleep/wake cycles or the keep_lora passes
MARGIN, PAD_S = 1.3, 300


def arm_s(arm, mode="tput", warm=0, measure=0):
    kind = arm.partition("-")[0]
    boot = BOOT_S.get(kind, SLOW_BOOT_S)
    if kind in ("R", "GR"):
        return boot
    if kind in SLEEP_PROBE_S:
        return boot + SLEEP_PROBE_S[kind]
    extra = DECODE_S
    if mode == "tput" and not re.fullmatch(r"J\d", kind):  # loaded arms: warm + measure per stream level
        levels = max(1, len([x for x in arm.partition("-")[2].split("+") if x]))
        extra += levels * (warm + measure + 60)
    return boot + extra


def need_s(arms, mode="tput", warm=150, measure=600):
    return int(MARGIN * (SETUP_S + sum(arm_s(a, mode, warm, measure) for a in arms.split(","))) + PAD_S)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("arms")
    p.add_argument("--mode", default="fe")
    p.add_argument("--warm", type=int, default=150)  # febench bench() defaults
    p.add_argument("--measure", type=int, default=600)
    p.add_argument("--cpu", action="store_true")
    a, _ = p.parse_known_args(argv)
    print(0 if a.cpu else need_s(a.arms, a.mode, a.warm, a.measure))


if __name__ == "__main__":
    main()
