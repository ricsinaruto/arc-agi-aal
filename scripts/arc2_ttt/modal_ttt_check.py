"""CPU check of ttt_core + arc2.ttt_data on a tiny random Qwen3 (the Qwen3 chat template, enable_thinking off) with a
PEFT LoRA, on 3 c240 tasks: labels cover exactly the answer, reset gives the base model back bit for bit, training
lowers the loss, read returns NLL/exact per query. Prints CHECK_RESULT. usage: modal run --detach modal_ttt_check.py"""
import json
from pathlib import Path

import modal
import modal.exception
try:  # grpclib's stream end on a dropped connection (pf11, 26 Oct)
    from grpclib.exceptions import StreamTerminatedError as _StreamTerminated
except ImportError:  # pragma: no cover
    _StreamTerminated = OSError

ROOT = Path(__file__).resolve().parents[min(2, len(Path(__file__).resolve().parents) - 1)]  # "/" in the container
DATA = ROOT / "dist/arc2-rtx-c240-hx-base18-v3b/data/arc2/evaluation"
TASKS = ["1818057f", "135a2760", "136b0064"]
app = modal.App("arc2-ttt-check")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.6.0", "transformers==4.51.3", "peft==0.15.2", "numpy==2.2.6",
                      extra_index_url="https://download.pytorch.org/whl/cpu")
         .add_local_dir(str(ROOT / "src/arc2"), "/w/arc2", ignore=["__pycache__"])
         .add_local_file(str(ROOT / "scripts/arc2_ttt/ttt_core.py"), "/w/ttt_core.py"))
for t in TASKS:
    image = image.add_local_file(str(DATA / f"{t}.json"), f"/w/tasks/{t}.json")


@app.function(image=image, cpu=(0.125, 8), timeout=1200, retries=0)
def check():
    import sys

    sys.path[:0] = ["/w"]
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    import ttt_core as tc
    from arc2 import ttt_data

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768)
    model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    checks, info = {}, {}
    for t in TASKS:
        task = json.loads(Path(f"/w/tasks/{t}.json").read_text())
        samples = ttt_data.samples(task, t, augments=2, max_tokens=6000)[:6]
        enc = [tc.encode(tok, s["messages"], s["target"]) for s in samples]
        enc = [e for e in enc if e is not None]
        ids, labels = enc[0]
        answer = tok.decode([i for i, l in zip(ids, labels) if l != -100])
        checks[f"{t}_label_is_answer"] = answer.startswith(samples[0]["target"]) and len(answer) - len(samples[0]["target"]) < 40
        checks[f"{t}_no_think_text_labelled"] = "<think>" not in answer
        q = [tc.encode(tok, x["messages"], x["target"]) for x in ttt_data.queries(task, t)]
        tc.reset(model, seed=1)
        base = tc.read(model, q)
        with model.disable_adapter():
            plain = tc.read(model, q)
        checks[f"{t}_reset_equals_base"] = [b["nll"] for b in base] == [p["nll"] for p in plain]
        before = [float(tc.loss_of(model, *e)) for e in enc]
        losses = tc.train(model, enc, epochs=4, lr=3e-3)
        after = [float(tc.loss_of(model, *e)) for e in enc]
        checks[f"{t}_train_lowers_loss"] = sum(after) < sum(before)
        adapted = tc.read(model, q)
        tc.reset(model, seed=1)
        again = tc.read(model, q)
        checks[f"{t}_reset_after_train"] = [b["nll"] for b in again] == [b["nll"] for b in base]
        info[t] = {"samples": len(enc), "tokens": [len(e[0]) for e in enc], "loss_first_last": [losses[0], losses[-1]],
                   "train_loss_before_after": [round(sum(before), 3), round(sum(after), 3)],
                   "query_base": base, "query_adapted": adapted}
    result = {"pass": all(checks.values()), "checks": checks, "info": info}
    print("CHECK_RESULT " + json.dumps(result), flush=True)
    return result

def wait_spawned(fn, *args):
    """spawn + wait, re-attaching by call id: a client holding .remote() cancels the call when its connection drops
    (r7/r8 26 Oct, pf-temp05b 27 Oct)."""
    import time as _time
    call = fn.spawn(*args)
    print("CALL " + call.object_id, flush=True)
    misses = 0
    while True:
        try:
            return call.get(timeout=60)
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except (OSError, modal.exception.ConnectionError, _StreamTerminated) as exc:  # a dropped connection: re-attach by id
            misses += 1
            if misses > 30:
                raise
            print("reattach after " + type(exc).__name__, flush=True)
            _time.sleep(20)
            call = modal.FunctionCall.from_id(call.object_id)


@app.local_entrypoint()
def main():
    print(json.dumps(wait_spawned(check))[:4000])
