#!/bin/bash
# vm_gone.sh RUN: exit 0 only if two instance listings 60 s apart both succeed and both lack arc-RUN (a failed CLI
# call is not "gone": rs_watch_run took a Nebius API error for the end of b2-sft4x-c1play-v1, 19 Oct)
for i in 1 2; do
  out=$(~/.nebius/bin/nebius compute instance list --parent-id project-e003l2wbt3gmofe9yxqr3y --all --format json 2>/dev/null) || exit 1
  echo "$out" | grep -q '"items"\|^\[' || exit 1
  echo "$out" | grep -qi "\"arc-$1\"" && exit 1  # names are lowercased (sAD read, 21 Oct)
  [ $i = 1 ] && sleep 60
done
exit 0
