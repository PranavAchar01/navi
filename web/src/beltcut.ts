// The 3D belt on its own, for the demo video: replays a take's logged belt messages in real
// time (?mode=replay) or runs the 4/8/16 layout tour (?mode=tour). Playback starts when the
// recorder calls window.__go(), which returns the wall-clock start in ms.
import { Belt3D } from "./belt3d";
import { cellsFromBelt } from "./overlay";
import type { Belt } from "./types";

type Cue = Pick<Belt, "duty" | "modes" | "lost"> & { t: number };

declare global {
  interface Window {
    __go: () => Promise<number>;
    __done: boolean;
  }
}

const mode = new URLSearchParams(location.search).get("mode") ?? "replay";
const belt = new Belt3D(document.getElementById("belt") as HTMLElement);
const { cues } = (await (await fetch("/_cut/belt-cues.json")).json()) as { cues: Cue[] };
window.__done = false;
const IDLE: Belt = {
  type: "belt",
  device: "replay",
  lost: false,
  drop: false,
  found: false,
  duty: [],
  modes: [],
  age_ms: null,
  latency_ms: null,
};

/** For the tour: one motor at a time, going round clockwise. */
function sweep(start: number) {
  const tick = () => {
    const k = Math.floor((performance.now() - start) / 450) % 8;
    const duty = Array.from({ length: 8 }, (_, i) => (i === k ? 255 : 0));
    const modes = Array.from({ length: 8 }, (_, i) => (i === k ? "solid" : "off"));
    belt.set(cellsFromBelt({ ...IDLE, duty, modes }));
    if (!window.__done) requestAnimationFrame(tick);
  };
  tick();
}

function replay(start: number) {
  const end = cues[cues.length - 1].t + 0.5;
  let i = -1;
  const tick = () => {
    const t = (performance.now() - start) / 1000;
    while (i + 1 < cues.length && cues[i + 1].t <= t) i++;
    if (i >= 0) belt.set(cellsFromBelt({ ...IDLE, ...cues[i] }));
    if (t < end) requestAnimationFrame(tick);
    else window.__done = true;
  };
  tick();
}

window.__go = async () => {
  const start = performance.now();
  if (mode === "tour") {
    belt.tour(3500);
    sweep(start);
    setTimeout(() => (window.__done = true), 4 * 3500);
  } else {
    replay(start);
  }
  return Date.now();
};
