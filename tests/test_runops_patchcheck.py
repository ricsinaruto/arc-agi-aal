"""Patch applicability: guards and edits checked against a pristine runtime tree (local site or runtime-cache parts)."""
import hashlib
import tarfile

from arc_infra.runops import patchcheck

GOOD = "def warm(self):\n    if self.mm_processor:\n        pass\n"


def bundle(tmp_path, sha_guard):
    src = tmp_path / "b/src/arc_infra"
    src.mkdir(parents=True)
    (src / "mm_patch.py").write_text(
        "RELATIVE = 'vllm/renderers/base.py'\n"
        "def patched_source(original):\n"
        "    t = original.decode()\n"
        "    if t.count('    if self.mm_processor:\\n') != 1:\n"
        "        raise RuntimeError('anchor is not unique')\n"
        "    return t.replace('if self.mm_processor:', 'if self.mm_processor and 0:').encode()\n")
    (src / "route_patch.py").write_text(f"TARGET = {{'relative': 'vllm/renderers/base.py', 'sha256': '{sha_guard}'}}\n")
    (src / "plain_patch.py").write_text("def apply(site, output, spec):\n    pass\n")
    return tmp_path / "b"


def site(tmp_path, text):
    f = tmp_path / "site/vllm/renderers/base.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return tmp_path / "site"


def test_matching_runtime_passes_and_stacking_is_reported(tmp_path):
    b = bundle(tmp_path, hashlib.sha256(GOOD.encode()).hexdigest())
    rep = patchcheck.check(b, site(tmp_path, GOOD))
    assert rep["ok"] and rep["patches"]["arc_infra/plain_patch"]["unchecked"]
    assert rep["stacked"] == {"vllm/renderers/base.py": ["arc_infra/mm_patch", "arc_infra/route_patch"]}


def test_a_changed_runtime_file_fails_the_gate(tmp_path):
    """v6f class: the image's file is not the one the patch was written for."""
    b = bundle(tmp_path, hashlib.sha256(GOOD.encode()).hexdigest())
    changed = GOOD.replace("pass", "self.warm_all()")
    rep = patchcheck.check(b, site(tmp_path, changed))
    assert not rep["ok"] and "!= guard" in rep["patches"]["arc_infra/route_patch"]["why"]
    rep = patchcheck.check(b, site(tmp_path, GOOD + "    if self.mm_processor:\n        pass\n"))
    assert "edit does not apply: anchor is not unique" in rep["patches"]["arc_infra/mm_patch"]["why"]
    assert patchcheck.main([str(b), "--site", str(tmp_path / "site")]) == 3


def test_reads_the_runtime_cache_parts(tmp_path):
    b = bundle(tmp_path, hashlib.sha256(GOOD.encode()).hexdigest())
    s = site(tmp_path, GOOD)
    key = tmp_path / "cache/key"
    key.mkdir(parents=True)
    with tarfile.open(key / "part-01.tar", "w") as t:
        t.add(s / "vllm/renderers/base.py", arcname="usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py")
    assert patchcheck.main([str(b), "--cache", str(key)]) == 0
