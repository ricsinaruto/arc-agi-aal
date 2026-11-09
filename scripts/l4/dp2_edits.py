"""build_b2 --dp2: edits on a bundle's copied bytes (hard links are replaced, never written through)."""
import shutil
from pathlib import Path

SERVING = [
    ('CARDS, CARD_NAME, ARCH, CONTEXT = 4, "L4", "8.9", 65536\n',
     'CARDS, CARD_NAME, ARCH, CONTEXT = 4, "L4", "8.9", 65536\nRANKS, REPLICAS = 2, 2  # l4kv dp2: two engines of two cards\n'),
    ('    page = 2 * (10240 // CARDS * (3 + int(draft)) + 48 // CARDS * 128 * 128)\n',
     '    page = 2 * (10240 // RANKS * (3 + int(draft)) + 48 // RANKS * 128 * 128)\n'),
    ('    if not 1 <= slots <= streams:\n', '    if not 1 <= slots <= streams * REPLICAS:  # streams: an engine\'s\n'),
    ('        argv[at] = str(CARDS)\n',
     '        argv[at] = str(RANKS)\n'
     '        argv += ["--data-parallel-size", str(REPLICAS), "--api-server-count", "1"]  # the default is one a replica\n'
     '        from arc2 import l4_dp2\n\n'
     '        print("ARC2_DP2 " + json.dumps(l4_dp2.apply(recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages", working)), flush=True)\n'),
]
TRIAL = [
    ('        self.client, self.gate, self.model = client, gate, model\n',
     '        self.client, self.gate, self.model = client, gate, model\n        self.route = None  # l4kv dp2: the engine\'s header\n'),
    ('await self.client.post("chat/completions", json=payload, timeout=self.left())',
     'await self.client.post("chat/completions", json=payload, timeout=self.left(), headers=self.route)'),
    ('''    transport = Transport(client, gate, model=listing.json()["data"][0]["id"], effort=study["reasoning_effort"],
                          temperature=float(study["temperature"]), deadline=deadline, prefill=spec.get("prefill"),
                          hold=spec.get("hold_attempt_blocks"), grow_wait=spec.get("grow_wait", GROW_WAIT),
                          reserve=spec.get("reply_reserve"), template=spec.get("chat_template_extra"))
''', '''    from . import l4_dp2  # l4kv dp2: one gate and one transport an engine; a replica's pool is the layout's

    transport, gate = l4_dp2.split(
        lambda part: Transport(client, part, model=listing.json()["data"][0]["id"], effort=study["reasoning_effort"],
                               temperature=float(study["temperature"]), deadline=deadline, prefill=spec.get("prefill"),
                               hold=spec.get("hold_attempt_blocks"), grow_wait=spec.get("grow_wait", GROW_WAIT),
                               reserve=spec.get("reply_reserve"), template=spec.get("chat_template_extra")),
        lambda share: KVGate(capacity, slots=share, budget=budget, log=gate.log), int(study["generation_slots"]))
'''),
    ('    out = {label: values.get(key) for label, key in ENGINE_METRICS.items()}\n',
     '    out = {label: values.get(key) for label, key in ENGINE_METRICS.items()}\n'
     '    from . import l4_dp2  # l4kv dp2: /metrics sums the engines; KV is the mean, with each engine\'s beside it\n\n'
     '    out["kv_replicas"] = l4_dp2.replica_kv(response.text)\n'
     '    if out["kv_replicas"]:\n'
     '        out["kv_cache_fraction"] = sum(out["kv_replicas"]) / len(out["kv_replicas"])\n'),
]
INT8 = [
    ('ROW_PARALLEL_K = 1536  # .linear_attn.out_proj, .self_attn.o_proj at TP 4\n',
     'ROW_PARALLEL_K = 3072  # .linear_attn.out_proj, .self_attn.o_proj at TP 2 (l4kv dp2)\n'),
    ('TILES = {(4096, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)), (2560, 1536): ((32, 256, 4, 3), (64, 128, 4, 3)),\n'
     '         (3584, 2560): ((16, 256, 4, 3), (64, 128, 4, 3)), (62080, 2560): ((32, 256, 4, 3), (32, 256, 4, 3))}\n',
     'TILES = {(8192, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)), (2560, 3072): ((32, 256, 4, 3), (64, 128, 4, 3)),\n'
     '         (7168, 2560): ((16, 256, 4, 3), (64, 128, 4, 3)), (124160, 2560): ((32, 256, 4, 3), (32, 256, 4, 3))}\n'),
]


ARENA = [
    ("    if type(sequences) is not int or sequences < study['generation_slots']:\n",
     "    if type(sequences) is not int or sequences * 2 < study['generation_slots']:  # l4kv dp2: an engine's\n"),
]


def edit(path, rows):
    text = path.read_text()
    path.unlink()  # a hard link to the base bundle's bytes
    for old, new in rows:
        assert text.count(old) == 1, f'{path.name}: {old[:70]!r}'
        text = text.replace(old, new)
    path.write_text(text)


def apply(out):
    """Serving, gate and INT8 edits on the bundle at ``out`` (after the other edits of those files)."""
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / 'l4_dp2_src.py', out / 'src/arc2/l4_dp2.py')
    edit(out / 'src/arc2/l4_serving.py', SERVING)
    edit(out / 'src/arc2/trial.py', TRIAL)
    edit(out / 'src/arena/serving.py', ARENA)
    if (out / 'src/arc2/l4_int8v9c.py').exists():
        edit(out / 'src/arc2/l4_int8v9c.py', INT8)
