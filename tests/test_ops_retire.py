"""tools/ops/retire.py: only ended (>48 h), unkept per-run volumes retire; the archive is verified per tar."""
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ops"))
import retire  # noqa: E402


def rec(**kw):
    end = kw.pop("end", time.time() - 50 * 3600)
    return {"run": "r", "state": "ended", "volume": "arc-run-r", "ledger": [{"row": {"end_epoch": end}}], **kw}


def test_refusals():
    assert retire.refusal("r", rec()) is None
    assert "shared" in retire.refusal("r", rec(volume=None))
    assert "keep" in retire.refusal("r", rec(keep="owner"))
    assert "not ended" in retire.refusal("r", rec(state="running"))
    assert "h ago" in retire.refusal("r", rec(end=time.time() - 3600))
    assert "retired already" in retire.refusal("r", rec(retired={"epoch": 1}))
    assert "no record" in retire.refusal("r", None)


def test_archive_tree_one_tar_per_folder_verified(tmp_path):
    src = tmp_path / "src"
    for rel in ("validation/r/feed.jsonl", "validation/r/state/s1/a.bin", "validation/r-attempt2/x",
                "validation/preflight/rP/leg.json", "compile-cache/k"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    m = retire.archive_tree(src, tmp_path / "arch")
    assert m["files"] == 5 and set(m["tars"]) == {"validation__r.tar.gz", "validation__r-attempt2.tar.gz",
                                                  "validation__preflight.tar.gz", "rest.tar.gz"}
    with tarfile.open(tmp_path / "arch/validation__r.tar.gz") as tf:
        assert sorted(tf.getnames()) == ["validation/r/feed.jsonl", "validation/r/state/s1/a.bin"]
    assert (tmp_path / "arch/manifest.json").is_file() and not list((tmp_path / "arch").glob("*.part"))
