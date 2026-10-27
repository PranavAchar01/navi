"""Stream a recorded video to the server like a phone would, then ask a question.

python scripts/replay.py data/walk30_720p.mp4 --ask "where is the remote"
"""

import argparse
import asyncio
import json
import struct
import time

import cv2
import websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--settle", type=float, default=12, help="seconds to wait after the video")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / a.fps))
    stats = {"safety": 0, "answers": [], "memory": None, "alerts": [], "fps": []}
    async with websockets.connect(a.url, max_size=None) as ws:

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    stats["chunks"] = stats.get("chunks", 0) + 1
                    continue
                d = json.loads(m)
                if d["type"] == "safety":
                    stats["safety"] += 1
                    stats["fps"].append(d["fps"])
                    stats["last"] = d
                elif d["type"] == "answer":
                    stats["answers"].append(d)
                elif d["type"] == "memory":
                    stats["memory"] = d
                elif d["type"] == "alert":
                    stats["alerts"].append(d)

        rd = asyncio.create_task(reader())
        t0, i, seq = time.time(), 0, 0
        while True:
            ok, frame = cap.read()
            if not ok or i / src_fps > a.seconds:
                break
            if i % step == 0:
                small = cv2.resize(frame, (640, 360))
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await ws.send(struct.pack("<dI", time.time() * 1000, seq) + jpg.tobytes())
                seq += 1
                await asyncio.sleep(max(0, t0 + (i + step) / src_fps - time.time()))
            i += 1
        await asyncio.sleep(a.settle)
        if a.ask:
            await ws.send(json.dumps({"type": "ask", "text": a.ask}))
            await asyncio.sleep(4)
        rd.cancel()
    f = stats["fps"]
    print("frames sent", seq, "safety replies", stats["safety"], "map chunks", stats.get("chunks"))
    print("server fps median", sorted(f)[len(f) // 2] if f else None)
    print(
        "memory",
        {k: stats["memory"].get(k) for k in ("window", "infer_ms", "map_points")}
        if stats["memory"]
        else None,
    )
    print("last safety", json.dumps(stats.get("last"))[:400])
    for ans in stats["answers"]:
        print("answer:", ans["text"], "| via", ans.get("language"))
    print("alerts", stats["alerts"])


asyncio.run(main())
