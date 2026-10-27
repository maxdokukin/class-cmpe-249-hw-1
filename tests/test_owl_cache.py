import numpy as np
import pytest

from hw1.data import iter_images, load_gt
from hw1.detectors.hf import OWLv2
from hw1.owl_cache import CachedOwl, build_cache

VOCAB = ["car", "van", "truck", "pedestrian", "cyclist", "tram"]


@pytest.mark.slow
def test_parity(tmp_path):
    path = build_cache("kitti_dev", tmp_path / "c.pt", max_images=3)
    cached = CachedOwl(path)
    det = OWLv2()
    det.set_vocabulary(VOCAB)
    gt = load_gt("kitti_dev")
    gt = {**gt, "images": gt["images"][:3]}
    for iid, img in iter_images(gt, "kitti_dev"):
        ref, got = det.predict(img), cached.predict(iid, VOCAB)
        n = min(20, len(ref["scores"]))
        assert ref["labels"][:n] == got["labels"][:n]
        np.testing.assert_allclose(got["scores"][:n], ref["scores"][:n], rtol=1e-3)
        assert np.abs(got["boxes"][:n] - ref["boxes"][:n]).max() < 1.0
