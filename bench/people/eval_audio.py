"""DoA and VAD on spoken sentences (macOS `say`), spatialised onto two microphones with a
reverberant room impulse response per ear. Prints the numbers that go in docs/people.md."""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metasight.audio_doa import HOP, SOUND_MPS, SR, Doa, Vad  # noqa: E402

TEXT = [
    "Hey, did you find the keys I left near the kitchen table this morning?",
    "I think we should take the left door, it leads straight to the garden.",
    "Could you hand me that blue bag when you get a moment, please.",
    "The meeting moved to three o'clock, so we have time for lunch first.",
]
VOICES = ["Samantha", "Daniel", "Karen", "Alex"]
SPACING = 0.14  # an iPhone-width pair; glasses temples are about 0.15
tmp = Path(tempfile.mkdtemp())


def speech(i):
    out = tmp / f"s{i}.wav"
    subprocess.run(["say", "-v", VOICES[i], "-o", str(tmp / f"s{i}.aiff"), TEXT[i]], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(tmp / f"s{i}.aiff"),
            "-ar",
            str(SR),
            "-ac",
            "1",
            str(out),
        ],
        check=True,
    )
    sr, x = wavfile.read(out)
    return x.astype(np.float32) / 32768


def rir(rng, rt60, n=int(0.4 * SR)):
    t = np.arange(n) / SR
    h = rng.normal(size=n) * np.exp(-6.9 * t / rt60)
    return h / np.linalg.norm(h)


def spatialise(x, deg, rng, rt60, snr_db, noise_kind):
    lag = SPACING / SOUND_MPS * math.sin(math.radians(deg)) * SR
    n = len(x)
    k = np.arange(n)
    left = np.interp(k - lag / 2, k, x, left=0, right=0)
    right = np.interp(k + lag / 2, k, x, left=0, right=0)
    if rt60 > 0:
        # direct path kept (first sample), tail independent per ear: a diffuse reverberant field
        tail = lambda: np.r_[0, rir(rng, rt60)[1:]]  # noqa: E731  direct path kept, tail per ear
        left = left + 0.5 * fftconvolve(left, tail())[:n]
        right = right + 0.5 * fftconvolve(right, tail())[:n]
    sig = np.stack([left, right], 1)
    p = float(np.mean(sig[np.abs(sig[:, 0]) > 0.02] ** 2))
    if noise_kind == "white":
        nz = rng.normal(size=sig.shape)
    else:  # diffuse low-pass noise, independent per mic (a fan or street hiss)
        nz = fftconvolve(rng.normal(size=sig.shape), np.ones((40, 1)) / 40, axes=0)[:n]
    nz *= math.sqrt(p / 10 ** (snr_db / 10) / max(float(np.mean(nz**2)), 1e-12))
    return (sig + nz).astype(np.float32)


def run_doa(x2, vad):
    d = Doa(SPACING)
    vad.reset()
    blk = SR // 4  # 250 ms blocks
    est = None
    for s in range(0, len(x2) - blk, blk):
        b = x2[s : s + blk]
        vad.push(b.mean(1))
        d.push(b, vad.active())
        est = d.bearing()
    return est


def main():
    rng = np.random.default_rng(0)
    clean = [speech(i) for i in range(len(TEXT))]
    vad = Vad()
    print("vad kind:", vad.kind)
    rows = []
    for rt60 in (0.0, 0.4):
        for snr in (20, 10, 0):
            for noise in ("white", "diffuse"):
                errs, miss = [], 0
                for deg in (-60, -40, -20, 0, 20, 40, 60):
                    for i, x in enumerate(clean):
                        est = run_doa(spatialise(x, deg, rng, rt60, snr, noise), vad)
                        if est is None:
                            miss += 1
                        else:
                            errs.append(abs(est - deg))
                n = 7 * len(clean)
                e = np.array(errs) if errs else np.array([np.nan])
                rows.append(
                    dict(
                        rt60=rt60,
                        snr=snr,
                        noise=noise,
                        n=n,
                        answered=n - miss,
                        med_err=round(float(np.median(e)), 1),
                        p90_err=round(float(np.percentile(e, 90)), 1),
                        within10=round(float(np.mean(e <= 10)) * len(e) / n, 2),
                    )
                )
    print("DOA", json.dumps(rows))

    # VAD: ground truth = frames where the clean sentence is above its own floor
    out = []
    for kind in ("silero", "energy"):
        for snr in (20, 10, 0):
            tp = fp = fn = tn = 0
            for x in clean:
                lead = np.zeros(SR)
                sig = np.r_[lead, x, lead, x[:SR], lead]
                truth_env = np.abs(sig)
                gt = np.array(
                    [truth_env[s : s + HOP].max() > 0.02 for s in range(0, len(sig) - HOP + 1, HOP)]
                )
                nz = rng.normal(size=len(sig)) * math.sqrt(
                    np.mean(x[np.abs(x) > 0.02] ** 2) / 10 ** (snr / 10)
                )
                v = Vad(model=(kind == "silero"))
                pred = []
                for s in range(0, len(sig) - HOP + 1, HOP):
                    v.push((sig[s : s + HOP] + nz[s : s + HOP]).astype(np.float32))
                    pred.append(v.prob >= v.threshold)
                pred = np.array(pred)
                tp += int(np.sum(pred & gt))
                fp += int(np.sum(pred & ~gt))
                fn += int(np.sum(~pred & gt))
                tn += int(np.sum(~pred & ~gt))
            out.append(
                dict(
                    vad=kind,
                    snr=snr,
                    recall=round(tp / (tp + fn), 3),
                    precision=round(tp / max(1, tp + fp), 3),
                    false_alarm_rate=round(fp / max(1, fp + tn), 3),
                )
            )
    print("VAD", json.dumps(out))


main()


def two_talkers():
    """Two people talking at once from -40 and +40 degrees: what does Doa say?"""
    rng = np.random.default_rng(3)
    a, b = speech(0), speech(1)
    n = min(len(a), len(b))
    x = spatialise(a[:n], -40, rng, 0.4, 20, "diffuse") + spatialise(
        b[:n], 40, rng, 0.4, 99, "diffuse"
    )
    print("TWO_TALKERS", run_doa(x, Vad()))


two_talkers()
