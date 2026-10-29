"""Stream a recorded conversation (video plus its audio track) to the server in real time and
print who it hears and sees: the speaker, the voice direction, and the turn cues.

python scripts/replay_talk.py clip.mp4 --seconds 20 [--spacing 0.14] [--mono]

A stereo audio track is sent as a two-microphone pair `spacing` metres apart, which only makes
sense when the two channels really are two microphones on the wearer (a phone or glasses
recording kept as raw channels). A mixed-down stereo track is not a microphone pair: use --mono.
"""

import argparse
import asyncio
import base64
import json
import struct
import subprocess
import time

import cv2
import websockets

SR = 16000
CHUNK_S = 0.1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--spacing", type=float, default=0.14)
    ap.add_argument("--mono", action="store_true")
    a = ap.parse_args()
    ch = 1 if a.mono else 2
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-t",
            str(a.seconds),
            "-i",
            a.video,
            "-vn",
            "-ar",
            str(SR),
            "-ac",
            str(ch),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    per = int(SR * CHUNK_S) * ch * 2
    cap = cv2.VideoCapture(a.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / a.fps))
    log = []
    async with websockets.connect(a.url, max_size=None) as ws:

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                d = json.loads(m)
                if d["type"] == "people":
                    log.append(d)
                elif d["type"] == "speaker_cue":
                    print(f"  cue: turn {d['turn']} ({d['bearing']} deg) motors {d['motors']}")

        rd = asyncio.create_task(reader())
        t0, seq, i, sent = time.time(), 0, 0, 0
        while i / src_fps < a.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            now = i / src_fps
            while sent * CHUNK_S <= now and sent * per < len(pcm):
                blk = pcm[sent * per : (sent + 1) * per]
                msg = {
                    "type": "audio",
                    "sr": SR,
                    "ch": ch,
                    "spacing": a.spacing,
                    "pcm": base64.b64encode(blk).decode(),
                }
                await ws.send(json.dumps(msg))
                sent += 1
            if i % step == 0:
                small = cv2.resize(frame, (640, 360))
                _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await ws.send(struct.pack("<dI", time.time() * 1000, seq) + jpg.tobytes())
                seq += 1
                await asyncio.sleep(max(0, t0 + (i + step) / src_fps - time.time()))
            i += 1
        await asyncio.sleep(1)
        rd.cancel()
    spk = [
        (e["t"], e["speaker"], e["doa"], e["evidence"]) for e in log if e["speaker"] or e["voice"]
    ]
    print(f"frames {seq}, people events {len(log)}, with voice {sum(e['voice'] for e in log)}")
    print("max people in view", max((len(e["tracks"]) for e in log), default=0))
    for row in spk[:: max(1, len(spk) // 12)]:
        print("  t={:.1f} speaker={} doa={} evidence={}".format(*row))


asyncio.run(main())
