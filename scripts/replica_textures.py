"""Cut the flat things in Pranav's bedroom (whiteboard, posters, the red-sun wave print, the
switch plate) out of his own footage and rectify them into textures for the Blender replica.

    .venv/bin/python scripts/replica_textures.py

Corners are picked by hand on the named frames (pixel coordinates in the 504 x 896 frame,
clockwise from top-left). Output: data/rooms/bedroom/replica/tex/*.png plus a contact sheet.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WALKS = ROOT / "data" / "walks"
OUT = ROOT / "data" / "rooms" / "bedroom" / "replica" / "tex"

# name: (frame, corners TL TR BR BL, output width px, width/height of the real thing)
CUTS = {
    "whiteboard": (
        "20261028-135040-iphone/frames/1793220651337-204.jpg",
        [(210, 204), (431, 204), (430, 354), (211, 354)],
        1024,
        0.825 / 0.556,
    ),
    "hoop_board": (
        "20261028-135040-iphone/frames/1793220651337-204.jpg",
        [(246, 112), (381, 112), (381, 198), (246, 198)],
        512,
        1.55,
    ),
    "wood_rail": (
        "20261028-135040-iphone/frames/1793220651337-204.jpg",
        [(214, 86), (406, 86), (406, 111), (214, 111)],
        1024,
        7.6,
    ),
    "switch_plate": (
        "20261028-135040-iphone/frames/1793220651337-204.jpg",
        [(228, 368), (279, 368), (279, 396), (228, 396)],
        256,
        1.8,
    ),
    "wave_print": (
        "20261102-023733-glasses/frames/1793615863907-52.jpg",
        [(47, 211), (180, 125), (246, 262), (92, 346)],
        768,
        0.68,
    ),
    "poster_collage": (
        "20261031-004732-glasses/frames/1793433208807-3419.jpg",
        [(238, 178), (378, 182), (377, 300), (238, 299)],
        768,
        1.2,
    ),
    "poster_comic": (
        "20261031-004732-glasses/frames/1793433208807-3419.jpg",
        [(386, 180), (470, 183), (469, 322), (387, 323)],
        512,
        0.62,
    ),
    "poster_purple": (
        "20261031-004732-glasses/frames/1793433208807-3419.jpg",
        [(237, 320), (385, 316), (383, 433), (243, 444)],
        768,
        1.25,
    ),
    "poster_mountain": (
        "20261031-004732-glasses/frames/1793433208807-3419.jpg",
        [(2, 170), (235, 177), (235, 340), (2, 334)],
        1024,
        1.45,
    ),
    "door_poster": (
        "20261029-182525-video/frames/008548.jpg",
        [(442, 130), (500, 128), (500, 228), (442, 226)],
        384,
        0.62,
    ),
}


def cut(frame: str, corners, width: int, aspect: float) -> np.ndarray:
    img = cv2.imread(str(WALKS / frame))
    h = int(round(width / aspect))
    src = np.float32(corners)
    dst = np.float32([(0, 0), (width - 1, 0), (width - 1, h - 1), (0, h - 1)])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (width, h), flags=cv2.INTER_LANCZOS4)
    # the footage is soft at this size; a light unsharp mask brings the print edges back
    blur = cv2.GaussianBlur(out, (0, 0), width / 400)
    return cv2.addWeighted(out, 1.5, blur, -0.5, 0)


def drawn() -> dict[str, np.ndarray]:
    """Things the footage only shows in passing, redrawn to match: the two lit screens, the
    three framed prints by the coat stand and the striped football shirt over the bed."""
    rng = np.random.default_rng(7)
    out = {}
    # curved monitor: a light app window over a dark editor, cyan side panel as in his frames
    s = np.full((1080, 2560, 3), (38, 33, 30), np.uint8)
    s[60:1020, 120:1500] = (236, 236, 238)
    s[60:110, 120:1500] = (214, 214, 218)
    cv2.rectangle(s, (420, 260), (980, 900), (222, 226, 232), -1)
    cv2.rectangle(s, (560, 420), (840, 700), (200, 214, 232), -1)
    s[60:1020, 1540:2440] = (40, 38, 46)
    s[60:1020, 2200:2440] = (226, 214, 120)
    for k in range(40):
        y = 140 + k * 21
        w = int(rng.integers(120, 560))
        col = [(170, 170, 180), (120, 200, 230), (200, 150, 230), (140, 220, 160)][k % 4]
        cv2.rectangle(s, (1590, y), (1590 + w, y + 9), col, -1)
    s[1030:1080] = (60, 52, 48)
    out["screen_curved"] = s
    # laptop: dark terminal
    t = np.full((1000, 1560, 3), (24, 22, 22), np.uint8)
    for k in range(44):
        y = 40 + k * 21
        w = int(rng.integers(80, 1300))
        col = (210, 210, 210) if k % 5 else (120, 210, 120)
        cv2.rectangle(t, (40, y), (40 + w, y + 8), col, -1)
    t[960:1000] = (52, 48, 46)
    out["screen_laptop"] = t
    # framed prints: pale pink/grey brush shapes, a dark print with a green line, a blue study
    a = np.full((900, 700, 3), (238, 236, 234), np.uint8)
    for k in range(6):
        cx = 140 + k * 85
        col = [(150, 120, 210), (120, 120, 130), (170, 150, 225), (90, 90, 100)][k % 4]
        cv2.ellipse(a, (cx, 450), (34, 230), 0, 0, 360, col, -1)
    out["print_a"] = a
    b = np.full((700, 900, 3), (28, 30, 30), np.uint8)
    cv2.rectangle(b, (90, 120), (560, 170), (120, 200, 120), -1)
    for k in range(9):
        cv2.rectangle(
            b,
            (90, 230 + k * 40),
            (90 + int(rng.integers(200, 700)), 250 + k * 40),
            (190, 190, 190),
            -1,
        )
    cv2.rectangle(b, (40, 40), (860, 660), (60, 150, 190), 6)
    out["print_b"] = b
    c = np.full((900, 700, 3), (236, 234, 230), np.uint8)
    for k in range(5):
        cv2.circle(c, (350, 450), 300 - k * 55, [(160, 90, 40), (60, 50, 200)][k % 2], 18)
    out["print_c"] = c
    # football shirt, front: garnet and blue stripes, gold 11, alpha cut to a T silhouette
    j = np.zeros((900, 900, 4), np.uint8)
    for k in range(9):
        col = (120, 40, 30, 255) if k % 2 else (40, 20, 130, 255)
        j[:, k * 100 : (k + 1) * 100] = col
    cv2.putText(j, "11", (300, 560), cv2.FONT_HERSHEY_DUPLEX, 8, (60, 200, 235, 255), 30)
    mask = np.zeros((900, 900), np.uint8)
    pts = np.array(
        [
            (250, 40),
            (370, 40),
            (450, 100),
            (530, 40),
            (650, 40),
            (880, 230),
            (790, 360),
            (700, 300),
            (700, 880),
            (200, 880),
            (200, 300),
            (110, 360),
            (20, 230),
        ],
        np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)
    j[..., 3] = np.minimum(j[..., 3], mask)
    out["jersey"] = j
    return out


# name: (frame, rect x0 y0 x1 y1): things with no flat outline, cut from the wall behind them
# with GrabCut and kept as RGBA cards; the builder places each card where the solved camera of
# that frame sees it (blender_replica_room.photo_card)
CARDS = {
    "door_garments": ("20261028-135040-iphone/frames/1793220651337-204.jpg", (0, 286, 142, 618)),
}


def card(frame: str, rect) -> np.ndarray:
    img = cv2.imread(str(WALKS / frame))
    x0, y0, x1, y1 = rect
    mask = np.zeros(img.shape[:2], np.uint8)
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, (x0, y0, x1 - x0, y1 - y0), bgm, fgm, 6, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alpha = cv2.GaussianBlur(cv2.erode(alpha, np.ones((2, 2), np.uint8)), (3, 3), 0)
    out = np.dstack([img, alpha])[y0:y1, x0:x1]
    return cv2.resize(out, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (frame, rect) in CARDS.items():
        cv2.imwrite(str(OUT / f"{name}.png"), card(frame, rect))
    for name, img in drawn().items():
        cv2.imwrite(str(OUT / f"{name}.png"), img)
    tiles = []
    for name, (frame, corners, width, aspect) in CUTS.items():
        img = cut(frame, corners, width, aspect)
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        t = cv2.resize(img, (256, min(420, int(256 * img.shape[0] / img.shape[1]))))
        tiles.append(cv2.copyMakeBorder(t, 0, 420 - t.shape[0], 0, 4, cv2.BORDER_CONSTANT))
    cv2.imwrite(str(OUT.parent / "tex-sheet.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    main()
