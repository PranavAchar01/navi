"""Small-object finding (the SAM 3.1 role): boxes for the watch-list in one keyframe.

`yoloe` is local open-vocabulary YOLOE-26 (free, weak on small things like keys). `sam31` is
SAM 3.1 on the Meta Model API: one request per concept, boxes parsed from the token stream.
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import cv2
import httpx
import numpy as np

from ..config import Settings, secret

# Furniture the pins can be described against ("near the dining table"). COCO class -> spoken name.
LANDMARKS = {
    "dining table": "dining table",
    "couch": "sofa",
    "chair": "chair",
    "bed": "bed",
    "tv": "TV",
    "refrigerator": "fridge",
    "oven": "oven",
    "sink": "sink",
    "microwave": "microwave",
    "toilet": "toilet",
    "laptop": "laptop",
    "potted plant": "plant",
}


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels


class Finder(Protocol):
    name: str

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]: ...


class YoloeFinder:
    name = "yoloe-26s-seg"

    def __init__(self, s: Settings, device: str = "mps", conf: float = 0.12):
        from ultralytics import YOLO

        self._model = YOLO(str(s.weights_dir / "yoloe-26s-seg.pt"))
        self._labels: list[str] | None = None
        self._device, self._conf = device, conf

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        if self._labels != labels:
            self._model.set_classes(labels)
            self._labels = list(labels)
        r = self._model.predict(bgr, device=self._device, conf=self._conf, verbose=False)[0]
        return [
            Detection(labels[int(c)], float(p), tuple(map(float, b)))
            for b, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            )
        ]


_BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([0-9.]+))?\|>")


def parse_sam_boxes(text: str, label: str, frame_w: int, frame_h: int) -> list[Detection]:
    """Boxes from SAM's special-token output, scaled from the reported w/h to our frame."""
    out = []
    for x1, y1, x2, y2, w, h, c in _BOX.findall(text):
        sx, sy = frame_w / int(w), frame_h / int(h)
        out.append(
            Detection(
                label,
                float(c) if c else 1.0,
                (int(x1) * sx, int(y1) * sy, (int(x2) + 1) * sx, (int(y2) + 1) * sy),
            )
        )
    return out


class Sam31Finder:
    name = "sam-3.1"

    def __init__(self, s: Settings, api_key: str, min_conf: float = 0.4):
        self._model, self._min_conf = s.sam_model, min_conf
        self._client = httpx.Client(
            base_url="https://api.meta.ai/v1",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"},
            timeout=30.0,
        )
        self._pool = ThreadPoolExecutor(max_workers=6)

    def _one(self, data_url: str, label: str, w: int, h: int) -> list[Detection]:
        body = {
            "model": self._model,
            "stream": True,
            "metadata": {"include_confidence": "true"},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": label},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        }
        text = ""
        with self._client.stream("POST", "/responses", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "response.output_text.delta":
                    text += ev.get("delta", "")
        return [d for d in parse_sam_boxes(text, label, w, h) if d.score >= self._min_conf]

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        h, w = bgr.shape[:2]
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        results = self._pool.map(lambda lb: self._one(url, lb, w, h), labels)
        return [d for r in results for d in r]


def make_finder(s: Settings) -> Finder:
    if s.finder == "sam31":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("finder = sam31 but no META_API_KEY / MODEL_API_KEY found")
        return Sam31Finder(s, key)
    return YoloeFinder(s)
