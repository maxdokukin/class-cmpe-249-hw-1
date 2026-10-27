from __future__ import annotations

import os
import resource
import sys
import time

import torch


def sync(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class Timer:
    """Per-image stage timer. mark() syncs the device, then records ms since the previous mark."""

    def __init__(self, device: str):
        self.device = device
        self.stages: dict[str, float] = {}
        self.start()

    def start(self) -> None:
        self.stages = {}
        sync(self.device)
        self._t0 = self._last = time.perf_counter()

    def mark(self, stage: str) -> None:
        sync(self.device)
        now = time.perf_counter()
        self.stages[stage + "_ms"] = (now - self._last) * 1e3
        self._last = now

    def result(self) -> dict:
        out = {"preprocess_ms": None, "forward_ms": None, "postprocess_ms": None}
        out.update(self.stages)
        out["total_ms"] = (self._last - self._t0) * 1e3
        return out


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        scale = 1 if sys.platform == "darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def _peak_rss_bytes() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


class MemSampler:
    """Call sample() after each image. mps: unified memory allocator footprints; cuda: tensor peak; cpu: process RSS."""

    def __init__(self, device: str):
        self.device = device
        if device.startswith("mps"):
            self.kind = "mps_unified"
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            self.kind = "cuda"
            torch.cuda.reset_peak_memory_stats()
        else:
            self.kind = "cpu_rss"
        self.peak = {"driver_bytes": 0, "current_bytes": 0, "rss_bytes": 0}
        self.n = 0

    def sample(self) -> None:
        sync(self.device)
        p = self.peak
        if self.kind == "mps_unified":
            p["driver_bytes"] = max(p["driver_bytes"], torch.mps.driver_allocated_memory())
            p["current_bytes"] = max(p["current_bytes"], torch.mps.current_allocated_memory())
        elif self.kind == "cuda":
            p["current_bytes"] = max(p["current_bytes"], torch.cuda.max_memory_allocated())
            p["driver_bytes"] = max(p["driver_bytes"], torch.cuda.max_memory_reserved())
        p["rss_bytes"] = max(p["rss_bytes"], _rss_bytes())
        self.n += 1

    def summary(self) -> dict:
        mb = lambda b: round(b / 2**20, 1)
        out = {"kind": self.kind, "samples": self.n, "process_rss_peak_mb": mb(max(self.peak["rss_bytes"], _peak_rss_bytes()))}
        if self.kind == "mps_unified":
            out.update(peak_driver_allocated_mb=mb(self.peak["driver_bytes"]), peak_current_allocated_mb=mb(self.peak["current_bytes"]))
        elif self.kind == "cuda":
            out.update(peak_allocated_mb=mb(self.peak["current_bytes"]), peak_reserved_mb=mb(self.peak["driver_bytes"]))
        else:
            out["peak_mb"] = out["process_rss_peak_mb"]
        return out
