from __future__ import annotations

from pathlib import Path

from .base import Detector

NAMES = ["yolo11m", "yolo26m", "frcnn_v2", "rtdetr_r50", "owlv2", "gdino_t", "fcos"]


def build_detector(name: str, device: str = "cpu", **overrides) -> Detector:
    if name in ("yolo11m", "yolo26m"):
        from .ultra import UltralyticsDetector

        if name == "yolo26m":
            overrides.setdefault("end2end", True)  # pass end2end=False for the NMS head
        wdir = Path(__file__).resolve().parents[3] / "weights"
        wdir.mkdir(exist_ok=True)
        return UltralyticsDetector(name, str(wdir / f"{name}.pt"), device=device, **overrides)
    if name in ("frcnn_v2", "fcos"):
        from .tv import TorchvisionDetector

        return TorchvisionDetector(name, device=device, **overrides)
    if name == "rtdetr_r50":
        from .hf import RTDetr

        return RTDetr(name, device=device, **overrides)
    if name == "owlv2":
        from .hf import OWLv2

        return OWLv2(name, device=device, **overrides)
    if name == "gdino_t":
        from .hf import GroundingDino

        return GroundingDino(name, device=device, **overrides)
    raise KeyError(f"unknown detector {name!r}; choose from {NAMES}")


__all__ = ["Detector", "build_detector", "NAMES"]
