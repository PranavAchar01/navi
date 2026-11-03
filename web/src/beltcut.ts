// The 3D belt on its own, for the demo video: replays a take's logged belt messages in real
// time (?mode=replay; ?take=ID reads _cut/belt-cues-ID.json), runs the 4/8/16 layout tour
// (?mode=tour), or turns the 8-motor belt round while each motor buzzes in turn (?mode=orbit).
// Playback starts when the recorder calls window.__go(), which returns the wall-clock start in ms.
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

const params = new URLSearchParams(location.search);
const mode = params.get("mode") ?? "replay";
const belt = new Belt3D(document.getElementById("belt") as HTMLElement);
const take = params.get("take")?.replace(/[^\w-]/g, "");
const cueFile = take ? `/_cut/belt-cues-${take}.json` : "/_cut/belt-cues.json";
const { cues } = (await (await fetch(cueFile)).json()) as { cues: Cue[] };
window.__done = false;
if (mode === "orbit") {
  belt.zoom = 1.3;
  belt.yaw = -0.45;
}
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
  if (mode === "orbit") {
    const len = 8000;
    const turn = () => {
      const u = Math.min(1, (performance.now() - start) / len);
      belt.yaw = -0.45 + 0.9 * (0.5 - 0.5 * Math.cos(Math.PI * u));
      if (u < 1) requestAnimationFrame(turn);
    };
    turn();
    sweep(start);
    setTimeout(() => (window.__done = true), len);
  } else if (mode === "tour") {
    belt.tour(3500);
    sweep(start);
    setTimeout(() => (window.__done = true), 4 * 3500);
  } else {
    replay(start);
  }
  return Date.now();
};
