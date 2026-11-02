// The 3D belt driven by a simulated walk (scripts/export_sim_scene.py writes the cues), for the
// sim render: ?cues=/_cut/sim/NAME.belt.json&layout=8. window.__go() replays the cues in real
// time for a live tab capture and returns the wall-clock start in ms; window.__show(t) holds
// the belt at one moment, for a still.
import { Belt3D } from "./belt3d";
import type { Layout } from "./beltlayout";
import { cellsFromBelt } from "./overlay";
import type { Belt } from "./types";

type Cue = Pick<Belt, "duty" | "modes" | "lost"> & { t: number };

declare global {
  interface Window {
    __go: () => Promise<number>;
    __show: (t: number) => void;
    __frames: () => number;
    __done: boolean;
  }
}

const q = new URLSearchParams(location.search);
const src = q.get("cues") ?? "";
if (!src.startsWith("/_cut/sim/")) throw new Error("cues must be under /_cut/sim/");
const belt = new Belt3D(document.getElementById("belt") as HTMLElement);
belt.setLayout(Number(q.get("layout") ?? 8) as Layout);
const { cues } = (await (await fetch(src)).json()) as { cues: Cue[] };
window.__done = false;
const IDLE: Belt = {
  type: "belt",
  device: "sim",
  lost: false,
  drop: false,
  found: false,
  duty: [],
  modes: [],
  age_ms: null,
  latency_ms: null,
};

function at(t: number): Cue {
  let i = 0;
  while (i + 1 < cues.length && cues[i + 1].t <= t) i++;
  return cues[i];
}

window.__show = (t: number) => belt.set(cellsFromBelt({ ...IDLE, ...at(t) }));
window.__frames = () => cues.length;

window.__go = async () => {
  const start = performance.now();
  const end = cues[cues.length - 1].t + 0.5;
  const tick = () => {
    const t = (performance.now() - start) / 1000;
    window.__show(t);
    if (t < end) requestAnimationFrame(tick);
    else window.__done = true;
  };
  tick();
  return Date.now();
};
