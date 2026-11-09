"""Host-memory guard (infra stage 3): periodic host-RAM readings into the metrics, and a check before each big host
allocation (a checkpoint copy, a teacher backlog, a weight load) that alerts when the container has less than
``margin`` (1.2) x the allocation available. The watcher reads the same keys (infra.host_avail_gib,
infra.next_alloc_gib) for its host-ram rule.

    guard = HostGuard(metrics_writer)          # or log=print without a writer
    guard.start()                              # a reading every 30 s
    if not guard.check(need_gib, "ckpt-12 host copy"):
        ...  # alerted; fall back (stream through the ring, smaller buffers) or stop cleanly
    guard.done()                               # the allocation is over

Each reading also exports hostmem.detail (top process RSS, pinned+locked, /dev/shm, /tmp, cgroup shmem/anon/file,
PSI). ``abort(reason)``: called once when two consecutive readings show the host below ``abort_gib`` available or
thrashing (PSI some avg10 >= 40% / full >= 10%), so the run ends itself (checkpoint, restart-guard end, exit) before
the kernel thrashes or the OOM killer takes it. Without ``abort`` the guard only alerts.
"""
import threading

from . import hostmem

MARGIN = 1.2


class HostGuard:
    def __init__(self, metrics=None, interval_s=30.0, margin=MARGIN, log=print, sample=None, detail=None,
                 abort=None, abort_gib=4.0):
        self.metrics, self.interval_s, self.margin, self.log = metrics, interval_s, margin, log
        self.sample_fn = sample or hostmem.sample
        self.detail_fn = detail if detail is not None else (hostmem.detail if sample is None else (lambda: {}))
        self.abort, self.abort_gib, self._bad, self.aborted = abort, abort_gib, 0, False
        self.last, self.min_avail, self.next_alloc = None, None, 0.0
        self._stop = threading.Event()

    def _emit(self, m, alert=None):
        if self.metrics is not None:
            self.metrics.emit(m, alert=alert)
        if alert and self.log:
            self.log(f"RUNOPS HOST-RAM {alert}")

    def read(self):
        s = self.sample_fn()
        self.last = s
        avail = s.get("avail_gib")
        if avail is not None and (self.min_avail is None or avail < self.min_avail):
            self.min_avail = avail
        try:
            d = self.detail_fn() or {}
        except Exception:  # noqa: BLE001 - detail is best effort
            d = {}
        self._emit({"infra.host_avail_gib": avail, "infra.host_rss_gib": s.get("rss_gib"),
                    "infra.host_cgroup_used_gib": s.get("cgroup_used_gib"), "infra.next_alloc_gib": self.next_alloc,
                    "infra.host_swap_gib": s.get("swap_used_gib"), **d})
        if hostmem.tight(s):
            self._emit({}, alert=f"host near exhaustion: {avail} GiB available of {s.get('total_gib')} GiB")
        bad = (avail is not None and avail < self.abort_gib) or hostmem.thrashing(d)
        self._bad = self._bad + 1 if bad else 0
        if self._bad >= 2 and self.abort and not self.aborted:
            self.aborted = True
            why = (f"host RAM abort: {avail} GiB available (< {self.abort_gib}), PSI some "
                   f"{d.get('infra.host_psi_some')} full {d.get('infra.host_psi_full')}, shm {d.get('infra.host_shm_gib')}"
                   f" GiB, pinned {d.get('infra.host_pinned_gib')} GiB")
            self._emit({}, alert=why)
            self.abort(why)
        return s

    def check(self, need_gib, what):
        """True when the host keeps ``margin`` x ``need_gib`` available; otherwise an alert and False."""
        self.next_alloc = float(need_gib)
        avail = self.read().get("avail_gib")
        if avail is None:
            return True  # no reading (not Linux): nothing to judge
        ok = avail >= self.margin * need_gib
        if not ok:
            self._emit({}, alert=f"{what} needs {need_gib:.1f} GiB host RAM, {avail:.1f} GiB available "
                                 f"(< {self.margin} x)")
        return ok

    def done(self):
        self.next_alloc = 0.0

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                self.read()
            except Exception as e:  # noqa: BLE001 - the guard never ends the run
                if self.log:
                    self.log(f"RUNOPS host guard read failed: {e!r}")

    def start(self):
        self.read()
        threading.Thread(target=self._loop, daemon=True, name="runops-hostguard").start()
        return self

    def stop(self):
        self._stop.set()
