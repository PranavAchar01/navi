"""The safety loop's depth net as a Core ML model for the phone's own reflex loop.

    uv run --with coremltools python scripts/coreml_depth.py

Depth Anything V2 Metric-Indoor Small (Apache-2.0), fixed at 294 x 518: the upright 9:16 frames
the app sends (504 x 896) map to that size in `align.map_resolution`, as on the server. The
ImageNet normalisation is baked into the graph, so the app hands over a plain BGRA pixel buffer.
Output `depth`: (1, 518, 294) float16, raw metric depth before the floor's scale correction.
Weights are stored as int8 (24 MB, against 47 MB at float16; median depth change 0.1 to 1% on
eight walk frames); the arithmetic stays float16. Writes
ios/CameraAccess/MetaSight/Models/ReflexDepth.mlpackage and, with --golden, the Swift port's
test frames: the model's depth and what Python's `analyse` and belt `plan` make of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

from metasight.safety import _MEAN, _STD, DEPTH_REPO, GRID, analyse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ios/CameraAccess/MetaSight/Models/ReflexDepth.mlpackage"
GOLDEN = ROOT / "ios/CameraAccessTests/Resources/reflex_golden.json"
W, H = 294, 518


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, rgb01):
        return self.net(pixel_values=(rgb01 - self.mean) / self.std).predicted_depth


def convert() -> None:
    net = AutoModelForDepthEstimation.from_pretrained(DEPTH_REPO, dtype=torch.float32).eval()
    # The input size is fixed, so the position embeddings' bicubic resize (no Core ML op) is done
    # once here and stored as a constant.
    emb = net.backbone.embeddings
    with torch.no_grad():
        dim = emb.position_embeddings.shape[-1]
        pos = emb.interpolate_pos_encoding(torch.zeros(1, 1, dim), H, W)
    emb.interpolate_pos_encoding = lambda e, h, w: pos
    # torch.export, not jit.trace: the trace bakes the decoder's resize sizes into int casts
    # that coremltools cannot read.
    exported = torch.export.export(Wrapped(net), (torch.rand(1, 3, H, W),)).run_decompositions({})
    ml = ct.convert(
        exported,
        inputs=[ct.ImageType("image", shape=(1, 3, H, W), scale=1 / 255, color_layout="RGB")],
        outputs=[ct.TensorType("depth")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    int8 = cto.OptimizationConfig(cto.OpLinearQuantizerConfig(mode="linear_symmetric"))
    ml = cto.linear_quantize_weights(ml, int8)
    ml.short_description = "Depth Anything V2 Metric-Indoor Small, 294x518, raw metric depth"
    ml.license = "Apache-2.0"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(OUT))
    print("wrote", OUT)


class _Recorder:
    """numpy's RANSAC draws, kept so the Swift port can replay the same triplets."""

    def __init__(self, seed):
        self.rng, self.draws = _default_rng(seed), []

    def choice(self, *a, **k):
        out = self.rng.choice(*a, **k)
        self.draws.append([int(x) for x in out])
        return out


_default_rng = np.random.default_rng


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def golden(frames: list[Path], eye_h: float, hfov: float, prior: float) -> None:
    """Each frame's depth (from the Core ML model, so only the geometry port is under test), what
    Python's `analyse` makes of it, and plan/packet cases for the belt command."""
    from PIL import Image

    from metasight.belt import packet, plan

    ml = ct.models.MLModel(str(OUT))
    cases = []
    for frame in frames:
        img = Image.open(frame).convert("RGB").resize((W, H), Image.BILINEAR)
        raw = np.asarray(ml.predict({"image": img})["depth"], np.float32)[0]
        # At the analysis grid already, so the test covers the geometry and not the resize.
        raw = cv2.resize(raw, GRID[::-1], interpolation=cv2.INTER_AREA).round(3)
        rec: list[_Recorder] = []
        np.random.default_rng = lambda seed, rec=rec: rec.append(_Recorder(seed)) or rec[-1]
        try:
            scene, plane = analyse(raw, prior, prior, hfov, eye_h, None)
        finally:
            np.random.default_rng = _default_rng
        cases.append(
            {
                "frame": frame.name,
                "width": raw.shape[1],
                "height": raw.shape[0],
                "depth": [round(float(x), 3) for x in raw.ravel()],
                "draws": rec[0].draws if rec else [],
                "sectors": [_finite(x) for x in scene.sectors],
                "path_m": _finite(scene.path_m),
                "drop_m": _finite(scene.drop_m),
                "floor": scene.floor,
                "plane": [*plane[0].tolist(), plane[1]] if plane is not None else None,
            }
        )
        print(frame.name, "floor", scene.floor, "sectors", scene.sectors, "path", scene.path_m)
    inf = float("inf")
    plans = []
    for sec, path, ttc, drop in [
        ([inf, 2.5, inf], inf, inf, inf),
        ([1.6, 0.8, 0.4], 0.7, inf, inf),
        ([inf, 1.9, inf], 1.9, 0.9, 1.2),
        ([0.3, inf, 1.2], 0.45, inf, 3.0),
    ]:
        merged = [inf] * 8
        merged[7], merged[0], merged[1] = sec  # front-left, front, front-right
        known = [i in (7, 0, 1) for i in range(8)]
        cmd, flags = plan(merged, known, path, ttc, drop)
        plans.append(
            {
                "sectors": [_finite(x) for x in sec],
                "path_m": _finite(path),
                "ttc_s": _finite(ttc),
                "drop_m": _finite(drop),
                "cmd": [list(c) for c in cmd],
                "flags": flags,
                "packet": packet(cmd, flags, 513, 1234567).hex(),
            }
        )
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(
        json.dumps({"eye_h": eye_h, "hfov": hfov, "scale": prior, "cases": cases, "plans": plans})
    )
    print("wrote", GOLDEN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, nargs="*", help="jpgs for the Swift test's golden")
    ap.add_argument("--eye-h", type=float, default=1.37)
    ap.add_argument("--hfov", type=float, default=105.0)
    ap.add_argument("--prior", type=float, default=0.45)
    a = ap.parse_args()
    if not OUT.exists():
        convert()
    if a.golden:
        golden(a.golden, a.eye_h, a.hfov, a.prior)


if __name__ == "__main__":
    main()
