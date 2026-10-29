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
    ap.add_argument("--onsets", default="", help="seconds where a voice starts, for latency")
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
    cues, belts, rtt, chunk_sent, frame_sent = [], [], [], {}, {}
    onsets = [float(x) for x in a.onsets.split(",") if x]
    async with websockets.connect(a.url, max_size=None) as ws:

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                d = json.loads(m)
                if d["type"] == "people":
                    log.append(d)
                elif d["type"] == "speaker_cue":
                    cues.append((time.time(), d))
                    print(f"  cue: turn {d['turn']} ({d['bearing']} deg) motors {d['motors']}")
                elif d["type"] == "belt":
                    belts.append((time.time(), d))
                elif d["type"] == "safety" and d["seq"] in frame_sent:
                    rtt.append(time.time() - frame_sent[d["seq"]])

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
                chunk_sent[sent] = time.time()
                await ws.send(json.dumps(msg))
                sent += 1
            if i % step == 0:
                small = cv2.resize(frame, (640, 360))
                _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await ws.send(struct.pack("<dI", time.time() * 1000, seq) + jpg.tobytes())
                frame_sent[seq] = time.time()
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

    print(
        f"frame round trip (send to safety reply): median "
        f"{1000 * sorted(rtt)[len(rtt) // 2]:.0f} ms"
        if rtt
        else "no safety replies"
    )
    print(f"cues {len(cues)}, belt messages {len(belts)}")
    if belts:
        print("  belt sample:", json.dumps(belts[len(belts) // 2][1])[:300])
    for on in onsets:
        t_on = chunk_sent.get(int(on / CHUNK_S))
        after = [c for c in cues if t_on and c[0] >= t_on]
        if after:
            c = after[0]
            print(
                f"voice at {on}s: first cue {1000 * (c[0] - t_on):.0f} ms after "
                "its audio was sent, "
                f"bearing {c[1]['bearing']}"
            )
        else:
            print(f"voice at {on}s: no cue")


asyncio.run(main())
