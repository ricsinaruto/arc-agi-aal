import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("l4_pp_src", ROOT / "scripts/l4/l4_pp_src.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
PINNED = Path(os.environ.get("L4PP_PINNED", "~/arc-shared/pinned-vllm"))


def fake_tree(tmp_path, files=None):
    for relative, edits in (files or pp.FILES).items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n" + "".join(old for old, _ in edits))


def test_apply_edits_once_and_records(tmp_path):
    fake_tree(tmp_path)
    record = pp.apply(tmp_path, tmp_path)
    assert all(row["changed"] for row in record.values()) and (tmp_path / "pp-patch.json").is_file()
    text = (tmp_path / "vllm/v1/worker/gpu_worker.py").read_text()
    assert "is_first_rank" in text and "unsupported.append" not in text
    assert not any(row["changed"] for row in pp.apply(tmp_path, tmp_path).values())


def test_draft_files_only_with_a_draft_and_a_moved_anchor_fails(tmp_path):
    fake_tree(tmp_path)  # MTP 0: the draft files are neither read nor needed
    assert set(pp.apply(tmp_path, tmp_path)) == set(pp.FILES)
    fake_tree(tmp_path / "d", {**pp.FILES, **pp.DRAFT_FILES})
    record = pp.apply(tmp_path / "d", tmp_path, draft=2)
    assert set(record) == {*pp.FILES, *pp.DRAFT_FILES} and all(row["changed"] for row in record.values())
    assert not any(row["changed"] for row in pp.apply(tmp_path / "d", tmp_path, draft=2).values())
    with pytest.raises(RuntimeError):
        pp.patched("nothing here", pp.FILES["vllm/v1/worker/gpu_worker.py"])
    with pytest.raises(RuntimeError):
        pp.patched("nothing here", pp.DRAFT_FILES["vllm/v1/worker/gpu/pp_utils.py"])


@pytest.mark.skipif(not PINNED.is_dir(), reason="needs the pinned vLLM tree")
def test_pinned_tree_patches_and_compiles():
    for relative, edits in {**pp.FILES, **pp.DRAFT_FILES}.items():
        source = (PINNED / relative).read_text()
        for old, _ in edits:
            assert source.count(old) == 1, (relative, old[:60])
        text = pp.patched(source, edits)
        assert pp.MARK in text and pp.patched(text, edits) == text
        compile(text, relative, "exec")


def test_partition_flag():
    spec = importlib.util.spec_from_file_location("pp_edits", ROOT / "scripts/l4/pp_edits.py")
    edits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edits)
    assert edits.partition("26, 22") == "26,22"
    for bad in ("24", "25,24", "0,48", "24,24,0"):
        with pytest.raises(ValueError):
            edits.partition(bad)
    assert edits.partition("22,19", 41) == "22,19"  # p50d7: 41 layers
    with pytest.raises(ValueError):
        edits.partition("24,24", 41)


def test_depth_cut_gate_edit():
    """build_b2's port of mkP50d7's NVFP4 PLE gate edit: one anchor, compiles, the checkpoint's layer types read."""
    spec = importlib.util.spec_from_file_location("build_b2", ROOT / "scripts/l4/build_b2.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    text = (ROOT / "src/arc_infra/serving_kernels_patch.py").read_text()
    if text.count(b.DEPTH_GATE[0]) != 1:
        pytest.skip("this tree's kernel patch carries no PLE gate edit")
    cut = text.replace(*b.DEPTH_GATE)
    compile(cut, "serving_kernels_patch.py", "exec")
    assert 'arc_unpruned_layer_types' in cut and cut.count('_RADIXARK_NVFP4_LAYER_TYPES') == 2


@pytest.mark.skipif(not (PINNED / "vllm/v1/worker/gpu/pp_utils.py").is_file(), reason="needs the pinned vLLM tree")
def test_drafts_reach_the_first_stage(monkeypatch):
    """Patched PPHandler, two stages in one process: a FIFO stands in for the NCCL broadcast."""
    import collections
    import sys
    import types
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    rank, channel = {"now": 1}, collections.deque()

    class Stream:
        def wait_stream(self, other): pass
        def wait_event(self, event): pass
        def record_event(self): return object()

    class Group:
        @property
        def is_last_rank(self): return rank["now"] == 1
        last_rank, world_size = 1, 2
        def make_sibling_device_group(self, group_desc): return group_desc

    def broadcast(tensor, src, group):
        if rank["now"] == src:
            channel.append(tensor.clone())
        else:
            tensor.copy_(channel.popleft())

    fakes = {"vllm": types.ModuleType("vllm"), "vllm.distributed": types.ModuleType("vllm.distributed"),
             "vllm.distributed.parallel_state": SimpleNamespace(get_pp_group=Group),
             "vllm.platforms": SimpleNamespace(current_platform=SimpleNamespace(is_xpu=lambda: False)),
             "vllm.v1.worker.gpu.buffer_utils": SimpleNamespace(
                 async_copy_to_gpu=lambda array, device=None: torch.from_numpy(np.array(array))),
             "vllm.v1.worker.gpu.input_batch": SimpleNamespace(InputBatch=object)}
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: __import__("contextlib").nullcontext())
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None, raising=False)
    relative = "vllm/v1/worker/gpu/pp_utils.py"
    module = types.ModuleType("pp_utils_patched")
    monkeypatch.setitem(sys.modules, module.__name__, module)  # the dataclass resolves its module
    exec(compile(pp.patched((PINNED / relative).read_text(), pp.DRAFT_FILES[relative]), relative, "exec"), module.__dict__)
    namespace = module.__dict__

    def batch(max_len=100):
        return SimpleNamespace(num_reqs=3, idx_mapping=torch.tensor([5, 2, 7]), idx_mapping_np=np.array([5, 2, 7]),
                               num_computed_tokens_np=np.array([10, 10, 10]), prefill_len_np=np.array([4, 4, 4]),
                               max_seq_len_np=np.array([max_len] * 3), num_scheduled_tokens=np.array([3, 3, 3]))

    def handlers(steps):
        rank["now"] = 1
        last = namespace["PPHandler"](8, steps, "cpu")
        rank["now"] = 0
        return last, namespace["PPHandler"](8, steps, "cpu")

    def step(last, first, b, drafts):
        rank["now"] = 1
        width = last.max_sample_len
        last.broadcast(torch.tensor([list(range(1, width + 1))] * 3), torch.tensor([width] * 3, dtype=torch.int32),
                       torch.tensor([0, 0, 0], dtype=torch.int32), b)
        last.broadcast_drafts(drafts, b)
        rank["now"] = 0
        assert first.get_prev_sampled_outputs() is None  # the seeded slot
        first.receive(b)
        assert not channel
        assert first.get_prev_sampled_outputs() is None
        return first.get_prev_sampled_outputs()  # pp_size steps later

    last, first = handlers(2)
    drafts = torch.tensor([[11, 12], [21, 22], [31, 32]])
    out = step(last, first, batch(), drafts)
    table = torch.zeros(8, 2, dtype=torch.int64)
    table[out["draft_tokens"][1]] = out["draft_tokens"][0]  # model_runner's write
    assert table[5].tolist() == [11, 12] and table[2].tolist() == [21, 22] and table[7].tolist() == [31, 32]

    last, first = handlers(2)
    rank["now"] = 1
    last.broadcast(torch.tensor([[1, 2, 3]] * 3), torch.tensor([3] * 3, dtype=torch.int32),
                   torch.tensor([0] * 3, dtype=torch.int32), batch())
    last.broadcast_drafts(drafts, batch())
    rank["now"] = 0
    first.get_prev_sampled_outputs()
    first.receive(batch())
    first.on_req_idx_freed(2)  # request index 2 ended before the consume
    first.get_prev_sampled_outputs()
    rows, idx = first.get_prev_sampled_outputs()["draft_tokens"]
    assert rows.tolist() == [[11, 12], [31, 32]] and idx.tolist() == [5, 7]

    last, first = handlers(2)  # every request finishing: nothing sent, nothing received, nothing left over
    rank["now"] = 1
    last.broadcast(torch.tensor([[1, 2, 3]] * 3), torch.tensor([3] * 3, dtype=torch.int32),
                   torch.tensor([0] * 3, dtype=torch.int32), batch(max_len=11))
    last.broadcast_drafts(drafts, batch(max_len=11))
    assert not channel
    rank["now"] = 0
    assert first.receive(batch(max_len=11)) is False

    last, first = handlers(0)  # MTP 0: the native two broadcasts only
    out = step(last, first, batch(), torch.empty(3, 0, dtype=torch.int64))
    assert out["draft_tokens"] is None


def test_serving_edits_match_the_source():
    spec = importlib.util.spec_from_file_location("pp_edits", ROOT / "scripts/l4/pp_edits.py")
    edits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edits)
    serving = (ROOT / "src/arc2/l4_serving.py").read_text()
    for old, _ in edits.SERVING:
        assert serving.count(old) == 1, old
    int8 = (ROOT / "scripts/l4/l4_int8v9c_src.py").read_text()
    for old, _ in edits.INT8:
        assert int8.count(old) == 1, old


def test_block_for_two_ranks_is_the_recipe_block():
    page = 2 * (10240 // 2 * 3 + 48 // 2 * 128 * 128)  # state_block_tokens at TP 2, MTP 0
    assert 64 * -(-page // (64 * 264)) == 3136


@pytest.mark.skipif(not (PINNED / "vllm/distributed/utils.py").is_file(), reason="needs the pinned vLLM tree")
def test_stage_split_env_and_the_offload_process(monkeypatch):
    """l4kd41: the PLE offload process (pp_size 1) inherited VLLM_PP_LAYER_PARTITION and vLLM raised; patched, it drops it."""
    import ast
    import types

    source = (PINNED / "vllm/distributed/utils.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "get_pp_indices")
    namespace = {"logger": types.SimpleNamespace(info=lambda *a: None)}

    class Envs:  # vLLM's envs reads the variable at each access
        def __getattr__(self, name):
            return os.environ.get(name)

    namespace["envs"] = Envs()
    exec(compile(ast.Module([node], []), "utils.py", "exec"), namespace)
    get_pp_indices = namespace["get_pp_indices"]
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "26,22")
    assert get_pp_indices(48, 0, 2) == (0, 26) and get_pp_indices(48, 1, 2) == (26, 48)
    with pytest.raises(ValueError, match="does not match pp_size=1"):
        get_pp_indices(48, 0, 1)  # the offload process before the fix
    relative = "vllm/v1/ple_offload/worker.py"
    text = pp.patched((PINNED / relative).read_text(), pp.FILES[relative])
    pop, build = text.index('environ.pop("VLLM_PP_LAYER_PARTITION"'), text.index("runner = PleOffloadRunner(vllm_config)")
    assert text.rindex("def proc_main", 0, pop) == text.rindex("def proc_main", 0, build) and pop < build
    lines = text[pop - 200:build].splitlines()
    snippet = "\n".join(line.strip() for line in lines if "_arc_os" in line or "cache_clear" in line)
    exec(snippet, {"envs": types.SimpleNamespace(__getattr__=lambda name: None)})  # the patched lines, as run
    assert "VLLM_PP_LAYER_PARTITION" not in os.environ and get_pp_indices(48, 0, 1) == (0, 48)
