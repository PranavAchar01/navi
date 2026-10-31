"""Keeps the room model current on its own: every finished walk is folded in.

A walk is finished when its frames.jsonl has not changed for SETTLE_S. For each one, in order:
  1. room_dense.py --start-from the current state, the new walk only (frames of other rooms
     do not fit the anchors and are dropped, so walking the whole house is fine);
  2. room_objects.py --walk: what moved since the last index, then the full index again;
  3. room_ledger.py: footage per room against the 10 hour benchmark;
  4. room_home.py: the home map the live app relocalizes against.
Each version lives in data/rooms/<room>/vNNN; data/rooms/<room>/current.json names the newest.
A failed step leaves current.json on the last good version and the walk is retried next round
(at most RETRIES times). Heavy steps take a heavy-lock slot and wait for memory.

    python scripts/room_watch.py --room bedroom --anchors $S/room160 --views 160 \
        --anchor-walks data/walks/20261031-004732-glasses
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WALKS = ROOT / "data" / "walks"
SETTLE_S = 600
RETRIES = 3
POLL_S = 120
PY = [str(ROOT / ".venv/bin/python")]
NICE = ["caffeinate", "-i", "-s", "nice", "-n", "19", "taskpolicy", "-b"]


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = next((ln for ln in out.splitlines() if "free percentage" in ln), "0%")
    return int(line.split()[-1].rstrip("%"))


def take_slot() -> Path:
    while free_pct() < 10 or shutil.disk_usage(ROOT).free < 15 * 2**30:
        time.sleep(30)
    n = 1
    while True:
        slot = ROOT.parent / (".heavy-lock" if n == 1 else f".heavy-lock-{n}")
        try:
            slot.mkdir()
        except FileExistsError:
            owner = slot / "owner"
            try:
                os.kill(int(owner.read_text()), 0)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                owner.unlink(missing_ok=True)  # stale: its owner is gone
                slot.rmdir()
                continue
            n += 1
            continue
        (slot / "owner").write_text(str(os.getpid()))
        return slot


def run(args: list[str], log: Path) -> bool:
    slot = take_slot()
    try:
        with log.open("a") as f:
            f.write(f"\n$ {' '.join(args)}\n")
            f.flush()
            return subprocess.run(NICE + PY + args, cwd=ROOT, stdout=f, stderr=f).returncode == 0
    finally:
        (slot / "owner").unlink(missing_ok=True)
        slot.rmdir()


def settled(walk: Path) -> bool:
    f = walk / "frames.jsonl"
    return f.is_file() and time.time() - f.stat().st_mtime > SETTLE_S


def save(sf: Path, state: dict) -> None:
    tmp = sf.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(sf)


def fold(walk: Path, rdir: Path, a, state: dict) -> bool:
    cur = rdir / state["current"] if state.get("current") else None
    # A fresh dir per attempt: room_dense appends to cams.jsonl, so reusing a failed attempt's
    # dir would count its cameras twice in the index and the ledger.
    n = len(state["versions"]) + 1
    while (ver := rdir / f"v{n:03d}").exists():
        n += 1
    ver.mkdir(parents=True)
    log = ver / "log.txt"
    start = ["--start-from", str(cur / "state.pt")] if cur else []
    common = ["--anchors", a.anchors, "--views", str(a.views), "--anchor-walks", *a.anchor_walks]
    ok = run(
        [
            "scripts/room_dense.py",
            str(walk),
            *common,
            *start,
            "--out",
            str(ver),
            "--strides",
            "2",
            "1",
            "--frame",
            str(rdir / "frame.json"),
        ],
        log,
    )
    tag = "s1"
    if ok and cur:  # the index of the version before is what this walk is compared against
        prev = cur / f"objects-{state['tag']}.json"
        if prev.is_file():
            (ver / f"objects-{tag}.json").write_text(prev.read_text())
        ok = run(
            ["scripts/room_objects.py", "--mesh", str(ver), "--tag", tag, "--walk", walk.name], log
        )
    if ok and cur:  # cams of earlier versions too, so the full index and ledger see every walk
        with (ver / "cams.jsonl").open("a") as f:
            f.write((cur / "cams.jsonl").read_text())
    ok = ok and run(["scripts/room_objects.py", "--mesh", str(ver), "--tag", tag], log)
    ok = ok and run(["scripts/room_ledger.py", "--mesh", str(ver), "--tag", tag], log)
    ok = ok and run(
        ["scripts/room_home.py", a.anchor_walks[0], "--anchors", a.anchors, "--views", str(a.views),
         "--mesh", str(ver), "--tag", tag, "--out", str(ROOT / "data" / "rooms" / a.room / "home")],
        log,
    )  # fmt: skip
    if ok:
        state["versions"].append(ver.name)
        state["current"], state["tag"] = ver.name, tag
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--views", type=int, required=True)
    ap.add_argument("--anchor-walks", nargs="+", required=True)
    ap.add_argument("--seed", help="a finished room_dense output dir to start from (its tag s1)")
    ap.add_argument("--once", action="store_true", help="one round, then exit")
    a = ap.parse_args()
    rdir = ROOT / "data" / "rooms" / a.room
    rdir.mkdir(parents=True, exist_ok=True)
    sf = rdir / "current.json"
    state = (
        json.loads(sf.read_text()) if sf.is_file() else {"versions": [], "done": {}, "tries": {}}
    )
    if a.seed and not state["versions"]:
        seed = Path(a.seed).resolve()
        state.update(versions=[str(seed)], current=str(seed), tag="s1")
        # Walks recorded from here on are folded; older ones are either in the seed or elsewhere.
        state["since"] = time.strftime("%Y%m%d-%H%M%S")
        (rdir / "frame.json").write_text((seed / "frame-s1.json").read_text())
        save(sf, state)  # now, or a restart would seed again and skip walks recorded meanwhile
    while True:
        for walk in sorted(WALKS.iterdir()):
            name = walk.name
            if (
                name < state.get("since", "")
                or name in state["done"]
                or state["tries"].get(name, 0) >= RETRIES
                or not settled(walk)
            ):
                continue
            print(f"{time.strftime('%H:%M')} folding {name}", flush=True)
            ok = fold(walk, rdir, a, state)
            if ok:
                state["done"][name] = state["current"]
            else:
                state["tries"][name] = state["tries"].get(name, 0) + 1
            save(sf, state)
            print(f"{time.strftime('%H:%M')} {name}: {'folded' if ok else 'failed'}", flush=True)
        if a.once:
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
