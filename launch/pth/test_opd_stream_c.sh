#!/bin/bash
# test_opd_stream_c.sh: quick test of opd_stream_c.sh's argument resolution: runs a copy of the launcher
# with stub launch_heal.sh / launch_rtx.sh / play_ready.py / jedit.py / pf_guard.sh that record their env, for the 3+1
# launch args, and checks what reaches launch_heal (preflight and round) and the play bundle's repoint.
set -u
L=./.claude/worktrees/sft-heal/launch/pth/opd_stream_c.sh; T=$(mktemp -d); mkdir -p $T/ops/edits $T/mon; fail=0
sed "s#^O=./.claude/worktrees/sft-heal/launch/pth; M=~/arc-sft-data/mon#O=$T/ops; M=$T/mon#" $L > $T/launcher.sh
grep -q "^O=$T/ops; M=$T/mon" $T/launcher.sh || { echo "FAIL: could not redirect O/M"; exit 1; }
cat > $T/ops/launch_heal.sh <<STUB
#!/bin/bash
f=$T/heal-\$NAME\${PREFLIGHT:+-pf}\${PARALLEL_PF:+-round}.env; env | sort > \$f
[ "\${PREFLIGHT:-}" = 1 ] && echo "PREFLIGHT CALL fc-test \$NAME" > $T/mon/preflight-sft-heal-round-\$NAME.log
[ "\${PARALLEL_PF:-}" = 1 ] && { echo "sft-heal-round-\$NAME ap-TESTTESTTESTTESTTESTTE 0 errors" | tee $T/mon/launch-sft-heal-round-\$NAME.log; echo "Merge with strategy ort failed."; }  # c3: a gpull merge failure as the last line
exit 0
STUB
printf '#!/bin/bash\necho "$@" > %s/rtx.args\n' $T > $T/ops/launch_rtx.sh
cat > $T/ops/play_ready.py <<'PY'
import sys  # stub: the c7b live play (g1-v15) is still booting
print(" ".join(sys.argv[1:])); sys.exit(1 if any("g1-v15" in a for a in sys.argv[1:]) else 0)
PY
printf 'pass\n' > $T/ops/jedit.py; printf '#!/bin/bash\nexit 0\n' > $T/ops/pf_guard.sh
chmod +x $T/ops/*.sh
cp ./.claude/worktrees/sft-heal/launch/pth/check_play_paths.py ./.claude/worktrees/sft-heal/launch/pth/prelaunch_check.py $T/ops/  # the real guards (c7, c7e 24 Oct)
(cd $T && HCARDS=4 HTRANKS=3 HRES=/runs/validation/sft-heal-round-sA-D-1e5-r3/train BAR_STEP=300 bash $T/launcher.sh > $T/out.txt 2>&1)
check() { grep -q "^$1=$2\$" $3 || { echo "FAIL: $1 != $2 in $(basename $3): $(grep "^$1=" $3)"; fail=1; }; }
for f in $T/heal-opdS-c4-pf-pf.env $T/heal-opdS-c4-round.env; do
  [ -f $f ] || { echo "FAIL: launch_heal not called ($(basename $f))"; fail=1; continue; }
  check CARDS 4 $f; check TRANKS 3 $f; check RESUME /runs/validation/sft-heal-round-sA-D-1e5-r3/train $f
  check BUNDLE dist/sft-heal-round-v64 $f; check TENV RL1_EXPERTS_FP8=1,RL1_STREAM_CHAINS=1 $f; check TSPREAD 1 $f
  check ARC_HEAL_STREAM_BAR_STEP 300 $f; check ARC_HEAL_STREAM_MAX_AGE 8 $f; check ARC_HEAL_STREAM_PLAY b2-sft4x-opdS-g1-v13 $f
  check LR 3.5e-6 $f; check BATCH 200000 $f; check SSTEPS 500 $f; check HTOTAL 500 $f; check GATE 0 $f; check TRUNC 0 $f; check ARC_HEAL_LOOP_SHARE 0 $f
  for kv in HEAL_DELTA_EVERY=1 HEAL_STREAM_BASE_SHARE=0.2 HEAL_SAMPLED_MODE=topk HEAL_MID_READ_EVERY=8 HEAL_STREAM_STALL=1800; do
    grep "^ARC_HEAL_ENV=" $f | tr , '\n' | grep -qx "\(ARC_HEAL_ENV=\)\?$kv" || { echo "FAIL: $kv not in ARC_HEAL_ENV ($(basename $f))"; fail=1; }; done
done
grep -q "^APP ap-TESTTESTTESTTESTTESTTE" $T/out.txt || { echo "FAIL: no APP line: $(tail -3 $T/out.txt)"; fail=1; }
[ -f $T/rtx.args ] && grep -q "b2-sft4x-opdS-g1-v13 .*sft4x-opdS-g1-v9" $T/rtx.args || { echo "FAIL: play launch args"; fail=1; }
python3 -I - <<PY || fail=1
import json
d = json.load(open("./.claude/worktrees/sft-prune/dist/sft4x-opdS-g1-v9/configs/comparison.json"))
want_in = "sft-heal-round-opdS-c4/deltas"
assert d["nebius_stream"]["in"] == want_in, d["nebius_stream"]
assert d["weights_follow"]["dir"] == f"/runs/validation/b2-sft4x-opdS-g1-v13/exchange-in/{want_in}", d["weights_follow"]
assert d["weights_follow"]["mode"] == "rows" and d["sampling"] == {"top_p": 1.0, "top_k": -1}, d["sampling"]
PY
# the play-bundle guard (c7, 24 Oct): a bundle whose stream-in / follower paths point at another round is refused
(cd $T && NAME=opdS-c7b PLAY=b2-sft4x-opdS-g1-v15 PB=sft4x-opdS-g1-v10 bash $T/launcher.sh > $T/out2.txt 2>&1)
grep -q "ALERT play bundle paths" $T/out2.txt || { echo "FAIL: guard did not refuse g1-v10 for c7b"; fail=1; }
# PLAY="LIVE,BACKLOG..." starts the round on the backlog without waiting on the live play (<45 s: one 20 s preflight poll, never the 60 s play poll)
(cd $T && NAME=opdS-c7b PLAY=b2-sft4x-opdS-g1-v15,b2-sft4x-opdS-g1-v8,b2-sft4x-opdS-g1-v14 PB=sft4x-opdS-g1-v11 SKIP_PLAY=1 bash $T/launcher.sh > $T/out3.txt 2>&1) &
w=$!; for i in $(seq 90); do kill -0 $w 2>/dev/null || break; sleep 0.5; done; kill $w 2>/dev/null && { echo "FAIL: backlog launch waited on the live play"; fail=1; }
f=$T/heal-opdS-c7b-round.env; [ -f $f ] && check ARC_HEAL_STREAM_PLAY b2-sft4x-opdS-g1-v15,b2-sft4x-opdS-g1-v8,b2-sft4x-opdS-g1-v14 $f && check ARC_HEAL_STREAM_PLAY_PENDING b2-sft4x-opdS-g1-v15 $f || { echo "FAIL: backlog round env"; fail=1; }
grep -q "play bundle sft4x-opdS-g1-v11 OK for b2-sft4x-opdS-g1-v15 " $T/out3.txt || { echo "FAIL: guard not run on the live play: $(head -2 $T/out3.txt)"; fail=1; }
[ $fail = 0 ] && echo "TEST OK opd_stream_c.sh (3+1 args, play repoint, bundle guard, backlog start)" || echo "TEST FAILED"; rm -rf $T; exit $fail
