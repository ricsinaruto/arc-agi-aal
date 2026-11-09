#!/bin/bash
# wait_par.sh "<run id> <dest folder>" ... : wait_copy.sh for each pair in parallel; prints as each lands
for pair in "$@"; do "$(dirname "$0")/wait_copy.sh" "$pair" & done; wait
