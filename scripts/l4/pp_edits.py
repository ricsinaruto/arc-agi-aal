"""build_b2 --pp2: edits on a bundle's copied bytes (hard links are replaced, never written through)."""
import shutil
from pathlib import Path

ENV = '        env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(CARDS)), "TORCH_CUDA_ARCH_LIST": ARCH}\n'
SERVING = [
    ('CARDS, CARD_NAME, ARCH, CONTEXT = 4, "L4", "8.9", 65536\n',
     'CARDS, CARD_NAME, ARCH, CONTEXT = 4, "L4", "8.9", 65536\nRANKS, STAGES = 2, 2  # l4pp: PP 2 + TP 2\n'
     'PP_PARTITION = None  # l4pp: VLLM_PP_LAYER_PARTITION, e.g. "26,22" (None: vLLM\'s even split)\n'),
    (ENV, ENV + '        if PP_PARTITION:  # l4pp: layers per stage (the draft sits on the last)\n'
     '            env["VLLM_PP_LAYER_PARTITION"] = PP_PARTITION\n'),
    ('    page = 2 * (10240 // CARDS * (3 + int(draft)) + 48 // CARDS * 128 * 128)\n',
     '    page = 2 * (10240 // RANKS * (3 + int(draft)) + 48 // RANKS * 128 * 128)\n'),
    ('        argv[at] = str(CARDS)\n',
     '        argv[at] = str(RANKS)\n'
     '        argv += ["--pipeline-parallel-size", str(STAGES)]\n'
     '        from arc2 import l4_pp\n\n'
     '        print("ARC2_PP " + json.dumps(l4_pp.apply(recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages", '
     'working, options.get("draft") or 0)), flush=True)\n'),
]
INT8 = [
    ('ROW_PARALLEL_K = 1536  # .linear_attn.out_proj, .self_attn.o_proj at TP 4\n',
     'ROW_PARALLEL_K = 3072  # .linear_attn.out_proj, .self_attn.o_proj at TP 2 (l4pp)\n'),
    ('TILES = {(4096, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)), (2560, 1536): ((32, 256, 4, 3), (64, 128, 4, 3)),\n'
     '         (3584, 2560): ((16, 256, 4, 3), (64, 128, 4, 3)), (62080, 2560): ((32, 256, 4, 3), (32, 256, 4, 3))}\n',
     'TILES = {(8192, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)), (2560, 3072): ((32, 256, 4, 3), (64, 128, 4, 3)),\n'
     '         (7168, 2560): ((16, 256, 4, 3), (64, 128, 4, 3)), (124160, 2560): ((32, 256, 4, 3), (32, 256, 4, 3))}\n'),
]


def edit(path, rows):
    text = path.read_text()
    path.unlink()  # a hard link to the base bundle's bytes
    for old, new in rows:
        assert text.count(old) == 1, f'{path.name}: {old[:70]!r}'
        text = text.replace(old, new)
    path.write_text(text)


def partition(text, total=48):
    """``"a,b"``: layers per stage, both >= 1, summing to the ``total`` decoder layers (41 on the d7 cut)."""
    parts = [int(x) for x in text.split(',')]
    if len(parts) != 2 or min(parts) < 1 or sum(parts) != total:
        raise ValueError(f'pp partition {text!r}: two stages, {total} layers')
    return f'{parts[0]},{parts[1]}'


def apply(out, layers=None, total=48):
    """Serving and INT8 edits on the bundle at ``out`` (after the other edits of those files); ``layers``: "a,b"."""
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / 'l4_pp_src.py', out / 'src/arc2/l4_pp.py')
    rows = SERVING
    if layers:
        rows = [(old, new.replace('PP_PARTITION = None', f'PP_PARTITION = {partition(layers, total)!r}')) for old, new in SERVING]
    edit(out / 'src/arc2/l4_serving.py', rows)
    if (out / 'src/arc2/l4_int8v9c.py').exists():
        edit(out / 'src/arc2/l4_int8v9c.py', INT8)
