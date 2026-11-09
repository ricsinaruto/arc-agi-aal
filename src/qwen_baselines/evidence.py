"""Durable observations, action receipts and incremental Python transcripts."""

import json
import sqlite3
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from arc_infra.perception import PALETTE

from . import objects as objects_


def png(grid, pixels=256):
    palette = np.array(
        [[int(c[i : i + 2], 16) for i in (0, 2, 4)] for c in PALETTE], dtype=np.uint8
    )
    image = Image.fromarray(palette[grid]).resize((pixels, pixels), Image.Resampling.NEAREST)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def grid_text(grid):
    return "\n".join("".join("0123456789ABCDEF"[int(c)] for c in row) for row in grid)


class Archive:
    def __init__(self, root, *, export=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.export = export
        self.cache = OrderedDict()
        self.objects_cache = OrderedDict()
        self.db = sqlite3.connect(self.root / "evidence.sqlite")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS frames(obs INTEGER, idx INTEGER, pixels BLOB NOT NULL,
                PRIMARY KEY(obs,idx));
            CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, code TEXT, metadata TEXT);
            CREATE TABLE IF NOT EXISTS output(call INTEGER, seq INTEGER PRIMARY KEY, text TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, kind TEXT, payload TEXT);
        """)

    def count(self, table="observations"):
        if table not in {"observations", "calls"}:
            raise ValueError("Unknown sequence")
        return self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def event(self, kind, **data):
        with self.db:
            self.db.execute(
                "INSERT INTO events(kind,payload) VALUES (?,?)", (kind, json.dumps(data))
            )

    def append(self, metadata, frames):
        index = self.count()
        meta = {
            **metadata,
            "observation_id": f"obs-{index:06d}",
            "index": index,
            "frame_count": len(frames),
            "intermediate_frame_count": max(0, len(frames) - 1),
            "settled_frame_index": len(frames) - 1,
        }
        refs = []
        for i, grid in enumerate(frames):
            stem = f"evidence/{meta['observation_id']}/frame-{i:03d}"
            ref = {
                "index": i,
                "role": "settled" if i == len(frames) - 1 else "intermediate",
                "grid_npy": stem + ".npy",
                "grid_text": stem + ".txt",
                "image_png": stem + ".png",
            }
            refs.append(ref)
            if self.export:
                for key in ("grid_npy", "grid_text", "image_png"):
                    ref[key] = str((self.root / ref[key]).resolve())
                (self.root / stem).parent.mkdir(parents=True, exist_ok=True)
                np.save(self.root / ref["grid_npy"], grid, allow_pickle=False)
                (self.root / ref["grid_text"]).write_text(grid_text(grid))
                (self.root / ref["image_png"]).write_bytes(png(grid))
            else:
                for key in ("grid_npy", "grid_text", "image_png"):
                    ref.pop(key)
        meta.update(
            frames=refs,
            settled=refs[-1] if refs else None,
            history_index=str((self.root / "evidence/index.jsonl").resolve())
            if self.export
            else "evidence/index.jsonl",
        )
        if not self.export:
            meta.pop("history_index")
        with self.db:
            self.db.execute("INSERT INTO observations VALUES (?,?)", (index, json.dumps(meta)))
            self.db.executemany(
                "INSERT INTO frames VALUES (?,?,?)",
                [(index, i, grid.tobytes()) for i, grid in enumerate(frames)],
            )
        if self.export:
            with (self.root / "evidence/index.jsonl").open("a") as stream:
                stream.write(json.dumps(meta) + "\n")
        return meta

    def metadata(self, index=-1):
        if index < 0:
            index += self.count()
        row = self.db.execute("SELECT metadata FROM observations WHERE id=?", (index,)).fetchone()
        if row is None:
            raise IndexError(index)
        return json.loads(row[0])

    def frame(self, obs=-1, index=-1):
        meta = self.metadata(obs)
        if index < 0:
            index += meta["frame_count"]
        key = (meta["index"], index)
        if key not in self.cache:
            row = self.db.execute("SELECT pixels FROM frames WHERE obs=? AND idx=?", key).fetchone()
            if row is None:
                raise IndexError(key)
            self.cache[key] = np.frombuffer(row[0], dtype=np.uint8).reshape(64, 64)
        self.cache.move_to_end(key)
        while len(self.cache) > 32:
            self.cache.popitem(last=False)
        return self.cache[key]

    def objects(self, index=-1):
        """Host-side segmentation of an observation's settled frame (cached, 8 entries)."""
        meta = self.metadata(index)
        key = meta["index"]
        if key not in self.objects_cache:
            self.objects_cache[key] = objects_.segment_host(self.frame(key))
        self.objects_cache.move_to_end(key)
        while len(self.objects_cache) > 8:
            self.objects_cache.popitem(last=False)
        return self.objects_cache[key]

    def object_view(self, index=-1, *, limit=40):
        return objects_.object_view(self.objects(index), limit=limit)

    def observation(self, index=-1):
        meta = self.metadata(index)
        return {
            **meta,
            "frames": [self.frame(meta["index"], i).tolist() for i in range(meta["frame_count"])],
            "grid": self.frame(meta["index"]).tolist(),
        }

    def transition(self, index):
        count = self.count() - 1
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(index)
        after = self.observation(index + 1)
        return {
            "action": after["action"],
            "before": self.observation(index),
            "after": after,
            "outcome": after["outcome"],
        }

    def begin_call(self, code):
        index = self.count("calls")
        with self.db:
            self.db.execute(
                "INSERT INTO calls VALUES (?,?,?)",
                (index, code, json.dumps({"status": "running", "before": self.count() - 1})),
            )
        return index

    def output(self, call, text):
        with self.db:
            self.db.execute("INSERT INTO output(call,text) VALUES (?,?)", (call, text))

    def end_call(self, call, **metadata):
        with self.db:
            prior = json.loads(
                self.db.execute("SELECT metadata FROM calls WHERE id=?", (call,)).fetchone()[0]
            )
            self.db.execute(
                "UPDATE calls SET metadata=? WHERE id=?",
                (json.dumps({**prior, **metadata, "after": self.count() - 1}), call),
            )

    def transcript(self, index):
        if index < 0:
            index += self.count("calls")
        row = self.db.execute("SELECT code,metadata FROM calls WHERE id=?", (index,)).fetchone()
        if row is None:
            raise IndexError(index)
        output = "".join(
            r[0]
            for r in self.db.execute("SELECT text FROM output WHERE call=? ORDER BY seq", (index,))
        )
        return {"call_id": index, "code": row[0], "output": output, **json.loads(row[1])}

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
