"""Small-object finding (the SAM 3.1 role): boxes for the watch-list in one keyframe.

`yoloe` is local open-vocabulary YOLOE-26 (free, weak on small things like keys). `sam31` is
SAM 3.1 on the Meta Model API: one request per concept, boxes parsed by meta-sam-parser.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from meta_sam_parser import image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI

from ..config import META_BASE_URL, Settings, secret

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


class Sam31Finder:
    """SAM 3.1 via the OpenAI SDK on the Meta Model API; `meta-sam-parser` reads the stream."""

    name = "sam-3.1"

    def __init__(self, s: Settings, api_key: str, min_conf: float = 0.4):
        self._model, self._min_conf = s.sam_model, min_conf
        self._api_key = api_key

    async def _one(self, client: AsyncOpenAI, data_url: str, label: str) -> list[Detection]:
        async with client.responses.stream(
            model=self._model,
            metadata={"include_confidence": "true"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": label},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        ) as stream:
            parsed = parse_responses_stream(stream, image_segmentation_format())
            result = await parsed.final_result()
        return [
            Detection(label, r.confidence if r.confidence is not None else 1.0, box)
            for r in result.records
            if r.kind == "box"
            and (box := (r.left, r.top, r.right, r.bottom))
            and (r.confidence is None or r.confidence >= self._min_conf)
        ]

    async def _all(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        # A fresh client per call: each find() runs in its own event loop.
        async with AsyncOpenAI(base_url=META_BASE_URL, api_key=self._api_key, timeout=30.0) as c:
            results = await asyncio.gather(*(self._one(c, url, lb) for lb in labels))
        return [d for r in results for d in r]

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        return asyncio.run(self._all(bgr, labels))


def make_finder(s: Settings) -> Finder:
    if s.finder == "sam31":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("finder = sam31 but no META_API_KEY / MODEL_API_KEY found")
        return Sam31Finder(s, key)
    return YoloeFinder(s)
