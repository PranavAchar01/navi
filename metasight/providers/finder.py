"""Small-object finding (the SAM 3.1 role): boxes for the watch-list in one keyframe.

`yoloe` is local open-vocabulary YOLOE-26 (free, weak on small things like keys). `sam31` is
SAM 3.1 on the Meta Model API: one request per concept, boxes parsed by meta-sam-parser.
`hybrid` finds everything locally and sends the small personal items (the things people lose)
to SAM 3.1: at once when the local model glimpses one, otherwise as a sweep at most once every
`meta_interval_s`. SAM is billed per image per concept, and a walk past a bottle on a desk can
show it for half a second, which a timed sweep alone misses.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
from meta_sam_parser import image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI

from ..config import META_BASE_URL, Settings, secret

log = logging.getLogger(__name__)

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
    source: str = ""


class Finder(Protocol):
    name: str

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]: ...


# What the open-vocabulary model is asked for, where the watch-list word reads it worse. On LVIS
# val (scripts/eval_finder.py) "cell phone" finds phones at AP50 0.63 where "phone" got 0.43
# (it also fires on desk phones, staplers and drills), and "key" beats "keys".
PROMPTS = {"phone": "cell phone", "keys": "key"}


class CropCheck:
    """A second look at each box: SigLIP 2 compares the crop with the item's name and with the
    names of ordinary things it is often mistaken for, and the detector's score is scaled by
    the square root of the item's share. On LVIS val this keeps most finds and drops most of
    the look-alikes (phone precision 0.54 to 0.79 at the 0.12 cut, wallet 0.35 to 0.54)."""

    MODEL = "google/siglip2-base-patch16-224"
    NAMES = {
        "keys": "a set of keys",
        "phone": "a smartphone",
        "remote control": "a TV remote control",
    }
    OTHER = (
        "a book", "a box", "a stapler", "a cup", "a mug", "a bag", "a laptop", "a keyboard",
        "a computer mouse", "a tray", "a toy", "a tool", "a pen", "a notebook", "a plate",
        "a bowl", "a shoe", "a pillow", "a picture frame", "a lamp", "a hand", "a person",
        "a chair", "a table", "a shelf", "a cable", "a can", "a jar", "a tablet", "a camera",
        "a clock", "a paper", "a towel", "a telephone", "food", "a wall",
    )  # fmt: skip

    def __init__(self, device: str = "mps"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch, self._device = torch, device
        self._proc = AutoProcessor.from_pretrained(self.MODEL)
        self._model = AutoModel.from_pretrained(self.MODEL).to(device).eval()
        self._labels: list[str] = []
        self._text = None

    def _features(self, out):
        f = getattr(out, "pooler_output", out)
        return f / f.norm(dim=-1, keepdim=True)

    def __call__(self, bgr: np.ndarray, dets: list[Detection], labels: list[str]):
        if not dets:
            return dets
        torch = self._torch
        if self._labels != labels:
            names = [self.NAMES.get(lb, f"a {lb}") for lb in labels] + list(self.OTHER)
            with torch.no_grad():
                t = self._proc(
                    text=[f"a photo of {n}" for n in names],
                    return_tensors="pt",
                    padding="max_length",
                ).to(self._device)
                self._text = self._features(self._model.get_text_features(**t))
            self._labels = list(labels)
        h, w = bgr.shape[:2]
        crops = []
        for d in dets:
            x0, y0, x1, y1 = d.box
            p = 0.15 * max(x1 - x0, y1 - y0)
            c = bgr[
                int(max(0, y0 - p)) : int(min(h, y1 + p)), int(max(0, x0 - p)) : int(min(w, x1 + p))
            ]
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            i = self._proc(images=crops, return_tensors="pt").to(self._device)
            f = self._features(self._model.get_image_features(**i))
            share = (f @ self._text.T * self._model.logit_scale.exp()).softmax(-1).float().cpu()
        return [
            replace(d, score=d.score * float(share[k, labels.index(d.label)]) ** 0.5)
            for k, d in enumerate(dets)
        ]


class YoloeFinder:
    """YOLOE-26 large at 1024 px, then the crop check. LVIS val AP50 over the watch-list: 0.29
    for the small model at 640, 0.44 for this one (about 165 ms a keyframe on the M5 Pro)."""

    name = "yoloe-26l-seg"

    def __init__(
        self,
        s: Settings,
        device: str = "mps",
        conf: float = 0.1,
        keep: float = 0.12,
        imgsz: int = 1024,
        check: bool = True,
    ):
        from ultralytics import YOLO

        self._model = YOLO(str(s.weights_dir / "yoloe-26l-seg.pt"))
        self._labels: list[str] | None = None
        self._device, self._conf, self._keep, self._imgsz = device, conf, keep, imgsz
        self._check = CropCheck(device) if check else None

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        if self._labels != labels:
            self._model.set_classes([PROMPTS.get(lb, lb) for lb in labels])
            self._labels = list(labels)
        r = self._model.predict(
            bgr, imgsz=self._imgsz, device=self._device, conf=self._conf, verbose=False
        )[0]
        dets = [
            Detection(labels[int(c)], float(p), tuple(map(float, b)), self.name)
            for b, c, p in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                strict=True,
            )
        ]
        if self._check is not None:
            dets = self._check(bgr, dets, labels)
        return [d for d in dets if d.score >= self._keep]


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
            Detection(label, r.confidence if r.confidence is not None else 1.0, box, self.name)
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


LOCAL_KEEP = 0.3  # a local hit on a personal item this sure stands even when SAM misses it
# What the local model calls a personal item when it is small or blurred: with the whole
# watch-list as its vocabulary, a water bottle on a desk mostly comes out as "cup".
LOOKALIKES = {
    "cup": ["water bottle"],
    "bottle": ["water bottle"],
    "remote control": ["phone"],
    "phone": ["remote control"],
    "handbag": ["wallet"],
    "book": ["wallet", "phone"],
}


class HybridFinder:
    """Everything on the Mac; small personal items confirmed and swept by SAM 3.1."""

    name = "yoloe + sam-3.1"

    def __init__(self, s: Settings, api_key: str):
        self.local, self.meta = YoloeFinder(s), Sam31Finder(s, api_key)
        self.items, self.interval = set(s.meta_items), s.meta_interval_s
        self.last_meta = 0.0
        self.meta_images = 0  # billed images so far this run

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        local = self.local.find(bgr, labels)
        glimpsed = sorted(
            {d.label for d in local if d.label in self.items}
            | {
                item
                for d in local
                for item in LOOKALIKES.get(d.label, [])
                if item in self.items and item in labels
            }
        )
        out = [d for d in local if d.label not in self.items]  # look-alikes stay as themselves
        now = time.monotonic()
        ask = glimpsed
        if now - self.last_meta >= self.interval:
            self.last_meta = now
            ask = [lb for lb in labels if lb in self.items]
        confirmed: list[Detection] = []
        if ask:
            self.meta_images += len(ask)
            try:
                confirmed = self.meta.find(bgr, ask)
            except Exception as e:  # the map keeps building on the local finder
                log.warning("SAM 3.1 failed: %s", e)
        have = {d.label for d in confirmed}
        keep = [d for d in local if d.label in self.items and d.label not in have]
        return out + confirmed + [d for d in keep if d.score >= LOCAL_KEEP]


def make_finder(s: Settings) -> Finder:
    if s.finder == "hybrid":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("finder = hybrid but no META_API_KEY / MODEL_API_KEY found")
        return HybridFinder(s, key)
    if s.finder == "sam31":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("finder = sam31 but no META_API_KEY / MODEL_API_KEY found")
        return Sam31Finder(s, key)
    return YoloeFinder(s)
