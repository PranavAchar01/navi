// The visual language (docs/VISUAL-DECISIONS.md): colour says what kind of thing it is,
// brightness says how much it matters right now. These mirror the tokens on :root in styles.css,
// for the canvas and WebGL renderers that cannot read CSS.

import type { Guide, PeopleMsg } from "./types";

export type RGB = [number, number, number];

export const DANGER: RGB = [239, 59, 59]; // obstacle under 0.9 m
export const CAUTION: RGB = [245, 165, 36]; // obstacle under 1.4 m
export const MEMORY: RGB = [0, 102, 204]; // something remembered: pins, recall answers
export const GUIDE: RGB = [18, 167, 118]; // the way to go: route ribbon, direction arc
export const PERSON: RGB = [219, 39, 119]; // a person or a voice, never who they are
export const YOU: RGB = [29, 29, 31]; // the wearer's position and heading

export const hex = ([r, g, b]: RGB) => (r << 16) | (g << 8) | b;
export const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r},${g},${b},${a.toFixed(3)})`;

/** Per-frame easing factor that settles in about `ms` (95 percent), whatever the frame rate. */
export function ease(dtMs: number, ms = 200): number {
  return 1 - Math.exp((-3 * Math.min(dtMs, 100)) / ms);
}

// A pin that answers a question pulses three times, then holds bright for 10 s.
const PULSE_MS = 700;
const HOLD_MS = 10_000;

/** How an answer pin looks `ms` after the answer: `ring` is the pulse phase (0..1, null once the
 *  three pulses are done) and `bright` is 1 while it holds, easing to 0 after. */
export function answerGlow(ms: number): { ring: number | null; bright: number } {
  if (ms < 0) return { ring: null, bright: 0 };
  const ring = ms < PULSE_MS * 3 ? (ms % PULSE_MS) / PULSE_MS : null;
  const end = PULSE_MS * 3 + HOLD_MS;
  return { ring, bright: ms < end ? 1 : Math.max(0, 1 - (ms - end) / 250) };
}

type V3 = readonly [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Degrees from straight ahead to `p`, positive to the right, flattened onto the floor. Right is
 *  fwd x up, which holds in any right-handed map frame (y up, or OpenCV's y down with up = -y),
 *  and matches metasight/heading.py. */
export function bearingDeg(pos: V3, fwd: V3, up: V3, p: V3): number {
  const n = Math.hypot(...up);
  const u: V3 = [up[0] / n, up[1] / n, up[2] / n];
  const d: V3 = [p[0] - pos[0], p[1] - pos[1], p[2] - pos[2]];
  const h = dot(d, u);
  const v: V3 = [d[0] - h * u[0], d[1] - h * u[1], d[2] - h * u[2]];
  const right: V3 = [fwd[1] * u[2] - fwd[2] * u[1], fwd[2] * u[0] - fwd[0] * u[2], fwd[0] * u[1] - fwd[1] * u[0]];
  return (Math.atan2(dot(v, right), dot(v, fwd)) * 180) / Math.PI;
}

/** The belt motor (index 0..7, clockwise from the front, as drawRing lays them out) nearest a bearing. */
export const motorAt = (bearing: number) => ((Math.round(bearing / 45) % 8) + 8) % 8;

/** The bearing to guide by: the server's own when it sends one, else the client's from the map. */
export function guideBearing(g: Guide | null | undefined, client: number | null): number | null {
  return g && typeof g.bearing_deg === "number" && Number.isFinite(g.bearing_deg) ? g.bearing_deg : client;
}

/** The server's people event (snake_case) as the bearings the edge arcs take. */
export function peopleBearings(m: PeopleMsg): { bearingDeg: number }[] {
  return (m.people ?? []).map((p) => ({ bearingDeg: p.bearing_deg })).filter((p) => Number.isFinite(p.bearingDeg));
}

/** Labels for people (the COCO "person" class and plain words for one). Decision 8: never marked. */
const PEOPLE = new Set(["person", "people", "man", "woman", "child", "boy", "girl"]);
export const isPerson = (label: string) => PEOPLE.has(label.trim().toLowerCase());
