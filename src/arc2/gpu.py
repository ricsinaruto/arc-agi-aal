"""GPU saturation record (eval, screen and training GPU runs record steady-state util and VRAM):
nvidia-smi every ``every`` seconds in a thread; ``summary()`` per phase and over the last ``recent`` seconds."""
import os
import subprocess
import threading
import time


class GpuSampler:
    def __init__(self, every=5.0, recent=600.0):
        self.every, self.recent = float(every), float(recent)
        self.rows, self.phase, self.stop = [], "start", threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def sample(self):
        out = subprocess.run([os.environ.get("ARC_REAL_NVIDIA_SMI", "nvidia-smi"),
                              "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [[float(v) for v in line.split(",")] for line in out.splitlines() if line.strip()]

    def run(self):
        while not self.stop.wait(self.every):
            try:
                cards = self.sample()
                self.rows.append((time.time(), self.phase, sum(c[0] for c in cards) / len(cards),
                                  sum(c[1] for c in cards) / 1024, sum(c[2] for c in cards) / 1024))
            except Exception:  # noqa: BLE001 - a missing sample is not an error
                pass
            if len(self.rows) > 200000:
                del self.rows[:100000]

    def mark(self, phase):
        self.phase = phase

    @staticmethod
    def stats(part):
        if not part:
            return None
        return {"samples": len(part), "util_mean": round(sum(r[2] for r in part) / len(part), 1),
                "idle_share": round(sum(r[2] < 10 for r in part) / len(part), 3),
                "memory_mean_gib": round(sum(r[3] for r in part) / len(part), 1),
                "memory_peak_gib": round(max(r[3] for r in part), 1), "memory_total_gib": round(part[-1][4], 1)}

    def summary(self):
        rows = list(self.rows)
        out = {phase: self.stats([r for r in rows if r[1] == phase]) for phase in sorted({r[1] for r in rows})}
        out["recent"] = self.stats([r for r in rows if r[0] >= time.time() - self.recent])
        out["all"] = self.stats(rows)
        return out
