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
    for the small model at 640, 0.46 for this one (about 165 ms a keyframe on the M5 Pro).
    With `tiles` = 2 it also looks at the four corners of the frame blown up to the same size, one
    batch with the full view, so small things get more pixels: 0.50, keys 0.11 to 0.15."""

    name = "yoloe-26l-seg"

    def __init__(
        self,
        s: Settings,
        device: str = "mps",
        conf: float = 0.1,
        keep: float = 0.12,
        imgsz: int = 1024,
        check: bool = True,
        tiles: int = 2,
    ):
        from ultralytics import YOLO

        self._model = YOLO(str(s.weights_dir / "yoloe-26l-seg.pt"))
        # The text prompts cost about 6 s to embed; do it here, not in the first window.
        self._model.set_classes([PROMPTS.get(lb, lb) for lb in s.watchlist])
        self._labels: list[str] | None = list(s.watchlist)
        self._device, self._conf, self._keep, self._imgsz = device, conf, keep, imgsz
        self._check = CropCheck(device) if check else None
        self._tiles = tiles

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        if self._labels != labels:
            self._model.set_classes([PROMPTS.get(lb, lb) for lb in labels])
            self._labels = list(labels)
        h, w = bgr.shape[:2]
        n = self._tiles
        th, tw = int(1.2 * h / max(n, 1)), int(1.2 * w / max(n, 1))  # a grid, a fifth overlap
        views = [(0, 0, bgr)]
        if n > 1:
            ys = np.linspace(0, h - th, n).astype(int)
            xs = np.linspace(0, w - tw, n).astype(int)
            views += [(x, y, bgr[y : y + th, x : x + tw]) for y in ys for x in xs]
        rs = self._model.predict(
            [v for _, _, v in views],
            imgsz=self._imgsz,
            device=self._device,
            conf=self._conf,
            verbose=False,
        )
        boxes, cls, conf = [], [], []
        for i, ((x, y, _), r) in enumerate(zip(views, rs, strict=True)):
            b = r.boxes.xyxy.cpu().numpy() + [x, y, x, y]
            if i:
                # a box cut by a tile's inner edge is a partial copy of something bigger
                e = 3
                ok = (
                    ((b[:, 0] > x + e) | (x == 0))
                    & ((b[:, 1] > y + e) | (y == 0))
                    & ((b[:, 2] < x + tw - e) | (x + tw >= w))
                    & ((b[:, 3] < y + th - e) | (y + th >= h))
                )
            else:
                ok = np.ones(len(b), bool)
            boxes.append(b[ok])
            cls.append(r.boxes.cls.cpu().numpy()[ok].astype(int))
            conf.append(r.boxes.conf.cpu().numpy()[ok])
        boxes, cls, conf = np.concatenate(boxes), np.concatenate(cls), np.concatenate(conf)
        if len(views) > 1 and len(boxes):
            import torch
            from torchvision.ops import batched_nms

            k = batched_nms(
                torch.from_numpy(boxes).float(),
                torch.from_numpy(conf).float(),
                torch.from_numpy(cls),
                0.5,
            ).numpy()
            boxes, cls, conf = boxes[k], cls[k], conf[k]
        dets = [
            Detection(labels[c], float(p), tuple(map(float, b)), self.name)
            for b, c, p in zip(boxes, cls, conf, strict=True)
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
        self.meta_images, self.meta_ms = 0, 0.0  # billed images so far, the last call's time

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

    async def _all(self, jobs: list[tuple[np.ndarray, list[str]]]) -> list[list[Detection]]:
        def url(img: np.ndarray) -> str:
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()

        calls = [(i, url(img), lb) for i, (img, labels) in enumerate(jobs) for lb in labels]
        # A fresh client per call: each find() runs in its own event loop.
        async with AsyncOpenAI(base_url=META_BASE_URL, api_key=self._api_key, timeout=30.0) as c:
            results = await asyncio.gather(*(self._one(c, u, lb) for _, u, lb in calls))
        out: list[list[Detection]] = [[] for _ in jobs]
        for (i, _, _), r in zip(calls, results, strict=True):
            out[i] += r
        return out

    def find_jobs(self, jobs: list[tuple[np.ndarray, list[str]]]) -> list[list[Detection]]:
        """Several images, each with its own labels, in one round of parallel calls."""
        t0 = time.perf_counter()
        out = asyncio.run(self._all(jobs))
        self.meta_images += sum(len(labels) for _, labels in jobs)
        self.meta_ms = 1000 * (time.perf_counter() - t0)
        return out

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        return self.find_jobs([(bgr, labels)])[0]


LOCAL_KEEP = 0.3  # a local hit on a personal item this sure stands even when SAM misses it
# SAM 3.1 is billed per image and concept, so it looks only for an item the wearer just asked
# for (and is being guided to), for this long; the house memory runs on the local model alone.
WANT_S = 90.0
# What leaves the Mac for SAM 3.1: a glimpsed item goes as a crop around the local hit (padded
# by the box's own size each side, at least CROP_MIN px), and the timed sweep as the whole frame
# shrunk to SWEEP_LONG_SIDE, enough for an item at arm's length but not to read papers or faces.
CROP_MIN = 256
SWEEP_LONG_SIDE = 640
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
    """Everything on the Mac; a small personal item the wearer asks for is confirmed and swept
    by SAM 3.1 until it is found or the ask runs out."""

    name = "yoloe + sam-3.1"

    def __init__(self, s: Settings, api_key: str):
        self.local, self.meta = YoloeFinder(s), Sam31Finder(s, api_key)
        self.items, self.interval = set(s.meta_items), s.meta_interval_s
        self.last_meta = 0.0
        self.meta_images = 0  # billed images so far this run
        self.meta_ms = 0.0  # how long the last SAM 3.1 call took, for the judge view
        self.wanted: dict[str, float] = {}  # item -> until when SAM 3.1 looks for it

    def want(self, label: str, seconds: float = WANT_S, new: bool = False) -> None:
        """Asked for, or being guided to: SAM 3.1 looks for this item from now for `seconds`.
        A `new` thing, one the house never saw, becomes a personal item: a weak local hit on it
        stands only once SAM 3.1 confirms it, so a word neither model can name pins nothing."""
        if new:
            self.items.add(label)
        if label in self.items:
            self.wanted[label] = time.monotonic() + seconds

    def find(self, bgr: np.ndarray, labels: list[str]) -> list[Detection]:
        local = self.local.find(bgr, labels)
        now = time.monotonic()
        wanted = {lb for lb, until in self.wanted.items() if until > now}
        glimpsed = sorted(
            {d.label for d in local if d.label in wanted}
            | {
                item
                for d in local
                for item in LOOKALIKES.get(d.label, [])
                if item in wanted and item in labels
            }
        )
        out = [d for d in local if d.label not in self.items]  # look-alikes stay as themselves
        ask = glimpsed
        if wanted and now - self.last_meta >= self.interval:
            self.last_meta = now
            ask = [lb for lb in labels if lb in wanted]
        confirmed: list[Detection] = []
        if ask:
            jobs = self._jobs(bgr, local, glimpsed, ask)
            self.meta_images += sum(len(labels) for _, labels, _ in jobs)
            try:
                t0 = time.perf_counter()
                found = self.meta.find_jobs([(img, labels) for img, labels, _ in jobs])
                self.meta_ms = 1000 * (time.perf_counter() - t0)
            except Exception as e:  # the map keeps building on the local finder
                log.warning("SAM 3.1 failed: %s", e)
                found = []
            for (_, _, (ox, oy, k)), dets in zip(jobs, found, strict=False):
                confirmed += [
                    replace(
                        d,
                        box=(
                            d.box[0] / k + ox,
                            d.box[1] / k + oy,
                            d.box[2] / k + ox,
                            d.box[3] / k + oy,
                        ),
                    )
                    for d in dets
                ]
        have = {d.label for d in confirmed}
        keep = [d for d in local if d.label in self.items and d.label not in have]
        return out + confirmed + [d for d in keep if d.score >= LOCAL_KEEP]

    def _jobs(self, bgr: np.ndarray, local: list[Detection], glimpsed: list[str], ask: list[str]):
        """(image, labels, (x offset, y offset, scale)) per SAM 3.1 request: a crop for each
        glimpsed item, one shrunk frame for the rest of a sweep."""
        h, w = bgr.shape[:2]
        jobs = []
        for label in glimpsed:
            boxes = [
                d.box for d in local if d.label == label or label in LOOKALIKES.get(d.label, ())
            ]
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            half_w = max((x1 - x0) * 1.5, CROP_MIN / 2)
            half_h = max((y1 - y0) * 1.5, CROP_MIN / 2)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x, t = int(max(0, cx - half_w)), int(max(0, cy - half_h))
            r, b = int(min(w, cx + half_w)), int(min(h, cy + half_h))
            jobs.append((bgr[t:b, x:r], [label], (x, t, 1.0)))
        rest = [lb for lb in ask if lb not in glimpsed]
        if rest:
            k = min(1.0, SWEEP_LONG_SIDE / max(h, w))
            small = (
                bgr if k == 1.0 else cv2.resize(bgr, (round(w * k), round(h * k)), cv2.INTER_AREA)
            )
            jobs.append((small, rest, (0, 0, k)))
        return jobs


def make_finder(s: Settings) -> Finder:
    if s.finder == "hybrid":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if key:
            return HybridFinder(s, key)
        # The local half alone still finds things; the map must not stop for a missing key.
        log.warning("finder = hybrid but no META_API_KEY / MODEL_API_KEY: using yoloe alone")
    if s.finder == "sam31":
        key = secret("META_API_KEY", "MODEL_API_KEY")
        if not key:
            raise RuntimeError("finder = sam31 but no META_API_KEY / MODEL_API_KEY found")
        return Sam31Finder(s, key)
    return YoloeFinder(s)
