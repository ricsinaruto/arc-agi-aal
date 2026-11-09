#!/bin/bash
# gpull.sh: safe pull of main in . (records.py pull: syncs and commits the run records and views,
# resolves view conflicts from the records, never leaves MERGE_HEAD behind)
exec ./.venv/bin/python ./tools/ops/records.py pull
