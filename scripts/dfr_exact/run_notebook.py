"""Execute a notebook top to bottom in an IPython kernel, as Kaggle does (dfranzen-exact, 4 October 2026).

    python run_notebook.py IN.ipynb OUT.ipynb [--cells 2,4,...] [--extra CODE]

Kaggle runs a notebook in an IPython kernel with the working directory /kaggle/working; nbclient does the same here
(cell magics, ``!`` shell lines and top-level ``await`` behave as on Kaggle). Stream output of the running cell is
echoed to stdout every few seconds (with the elapsed time) and the partial notebook is saved, so a Modal run has live
logs. ``--cells`` restricts execution to the given cell indices (the CPU self-test); ``--extra`` appends one code cell.
"""

import argparse
import copy
import sys
import threading
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--cells", default="")
    parser.add_argument("--extra", default="")
    parser.add_argument("--cwd", default="/kaggle/working")
    args = parser.parse_args()
    nb = nbformat.read(args.source, as_version=4)
    if args.cells:
        keep = [int(i) for i in args.cells.split(",")]
        nb.cells = [nb.cells[i] for i in keep]
    if args.extra:
        nb.cells.append(nbformat.v4.new_code_cell(args.extra))
    started = time.time()
    state = {"cell": None, "printed": {}}
    stop = threading.Event()

    def echo():
        index = state["cell"]
        if index is None:
            return
        cell = nb.cells[index]
        texts = []
        for output in list(cell.get("outputs", [])):
            if output.get("output_type") == "stream":
                texts.append(output.get("text", ""))
            elif output.get("output_type") == "error":
                texts.append("\n".join(output.get("traceback", [])) + "\n")
        text = "".join(texts)
        done = state["printed"].get(index, 0)
        if len(text) > done:
            for line in text[done:].splitlines():
                print(f"[nb {time.time() - started:7.0f}s cell {index}] {line}", flush=True)
            state["printed"][index] = len(text)

    def saver():
        while not stop.wait(5):
            try:
                echo()
            except Exception:
                pass
        echo()

    def on_start(cell, cell_index, **_):
        echo()
        state["cell"] = cell_index
        print(f"[nb {time.time() - started:7.0f}s] cell {cell_index} start", flush=True)

    def on_done(cell, cell_index, **_):
        echo()
        print(f"[nb {time.time() - started:7.0f}s] cell {cell_index} done", flush=True)
        try:
            nbformat.write(copy.deepcopy(nb), args.output)
        except Exception as exc:
            print(f"[nb] partial save failed: {exc}", flush=True)

    client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": args.cwd}},
                            on_cell_start=on_start, on_cell_executed=on_done)
    thread = threading.Thread(target=saver, daemon=True)
    thread.start()
    code = 0
    try:
        client.execute()
    except Exception as exc:
        print(f"[nb] execution stopped: {type(exc).__name__}: {str(exc)[-4000:]}", flush=True)
        code = 1
    finally:
        stop.set()
        thread.join(timeout=10)
        nbformat.write(nb, args.output)
    sys.exit(code)


if __name__ == "__main__":
    main()
