import type { Cell } from "./beltlayout";
import { type Belt, type Hazard, SECTOR_NAMES, type Safety } from "./types";
import { CAUTION, DANGER, GUIDE, MEMORY, PERSON, type RGB, ease, rgba } from "./visual";

// Hazards: only the obstacles in the walking path, nearest first, each a smooth thin contour
// with a soft inner fade and a small distance tag. Amber when you are approaching (under 1.4 m),
// red when you are too close (under 0.9 m). Hysteresis on the level, easing on colour and alpha.
// A blob that covers more than a quarter of the frame lets its fill go and keeps only the rim, so
// a close wall never paints the screen red.
const AMBER = CAUTION;
const RED = DANGER;
const RIM_ONLY = 0.25;
const RIM_BACK = 0.2; // the fill comes back only under this, so a blob at the threshold does not flicker
const SHAPE_N = 64; // contour points after resampling, so one outline can ease into the next
const SHAPE_MS = 120; // a new outline settles in about this long, instead of stepping per message
interface Look {
  level: number;
  rgb: [number, number, number];
  alpha: number;
  fill: number;
  rim: boolean;
  shape: number[] | null;
  t: number;
  seen: number;
  x: number;
  y: number;
}
let looks: Look[] = [];

function look(h: Hazard, now: number): Look {
  let best: Look | null = null;
  let bd = 0.15;
  for (const l of looks) {
    const dd = Math.hypot(l.x - h.tag[0], l.y - h.tag[1]);
    if (dd < bd && l.seen !== now) {
      best = l;
      bd = dd;
    }
  }
  if (!best) {
    best = { level: 1, rgb: [...AMBER], alpha: 0, fill: 1, rim: false, shape: null, t: now, seen: now, x: h.tag[0], y: h.tag[1] };
    looks.push(best);
  }
  best.x += (h.tag[0] - best.x) * 0.5;
  best.y += (h.tag[1] - best.y) * 0.5;
  const want = resample(h.poly, best.shape);
  if (!best.shape) best.shape = want;
  else {
    const k = ease(now - best.t, SHAPE_MS);
    for (let i = 0; i < want.length; i++) best.shape[i] += (want[i] - best.shape[i]) * k;
  }
  best.t = now;
  best.seen = now;
  if (best.level === 1 && h.dist < 0.9) best.level = 2;
  else if (best.level === 2 && h.dist > 1.05) best.level = 1;
  const target = best.level === 2 ? RED : AMBER;
  for (let i = 0; i < 3; i++) best.rgb[i] += (target[i] - best.rgb[i]) * 0.2;
  best.alpha += (1 - best.alpha) * 0.25;
  const c = cover(h.poly);
  if (c > RIM_ONLY) best.rim = true;
  else if (c < RIM_BACK) best.rim = false;
  best.fill += ((best.rim ? 0 : 1) - best.fill) * 0.2;
  return best;
}

/** A closed outline as SHAPE_N points evenly spaced along it, wound one way and started where it
 *  lines up best with `prev`, so the two can be blended point by point. */
export function resample(p: number[], prev: number[] | null): number[] {
  const n = p.length / 2;
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) pts.push([p[i * 2], p[i * 2 + 1]]);
  let a = 0;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    a += x0 * y1 - x1 * y0;
  }
  if (a < 0) pts.reverse();
  const len: number[] = [0];
  for (let i = 0; i < n; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    len.push(len[i] + Math.hypot(x1 - x0, y1 - y0));
  }
  const total = len[n] || 1;
  const out: number[] = [];
  for (let j = 0, i = 0; j < SHAPE_N; j++) {
    const d = (j / SHAPE_N) * total;
    while (i < n - 1 && len[i + 1] < d) i++;
    const f = (d - len[i]) / (len[i + 1] - len[i] || 1);
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    out.push(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f);
  }
  if (!prev) return out;
  let best = 0;
  let bd = Infinity;
  for (let r = 0; r < SHAPE_N; r++) {
    let d = 0;
    for (let j = 0; j < SHAPE_N; j++) {
      const k = ((j + r) % SHAPE_N) * 2;
      d += Math.abs(out[k] - prev[j * 2]) + Math.abs(out[k + 1] - prev[j * 2 + 1]);
    }
    if (d < bd) [best, bd] = [r, d];
  }
  const rot: number[] = [];
  for (let j = 0; j < SHAPE_N; j++) rot.push(out[((j + best) % SHAPE_N) * 2], out[((j + best) % SHAPE_N) * 2 + 1]);
  return rot;
}

/** Share of the frame inside a normalised outline (shoelace). */
function cover(p: number[]): number {
  let a = 0;
  const n = p.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    a += p[i * 2] * p[j * 2 + 1] - p[j * 2] * p[i * 2 + 1];
  }
  return Math.abs(a) / 2;
}

function smoothPath(ctx: CanvasRenderingContext2D, p: number[], ox: number, oy: number, w: number, h: number) {
  const n = p.length / 2;
  const X = (i: number) => ox + p[(i % n) * 2] * w;
  const Y = (i: number) => oy + p[(i % n) * 2 + 1] * h;
  ctx.beginPath();
  ctx.moveTo((X(0) + X(1)) / 2, (Y(0) + Y(1)) / 2);
  for (let i = 1; i <= n; i++) ctx.quadraticCurveTo(X(i), Y(i), (X(i) + X(i + 1)) / 2, (Y(i) + Y(i + 1)) / 2);
  ctx.closePath();
}

// The rim is drawn in runs of points with one strength each. Points on the picture border, or on a
// straight stretch spanning more than LONG_CUT of the picture width, get none: the edge of the picture
// (or a cut in the server's outline) is not the edge of the object. Strength comes back over
// RIM_FADE of the longer picture side, so the rim fades toward those ends instead of stopping hard.
const LONG_CUT = 0.35;
const RIM_FADE = 0.12;
const RIM_LEVELS = 5;

/** Runs of consecutive contour points (closed, normalised) and the rim strength (0..1) of each. */
export function rimRuns(p: number[], w: number, h: number): { a: number; idx: number[] }[] {
  const n = p.length / 2;
  const X = (i: number) => p[(((i % n) + n) % n) * 2] * w;
  const Y = (i: number) => p[(((i % n) + n) % n) * 2 + 1] * h;
  const fadeOf = Math.max(w, h);
  const off = new Array<boolean>(n).fill(false);
  for (let i = 0; i < n; i++) {
    const x = p[i * 2];
    const y = p[i * 2 + 1];
    if (x < 0.006 || x > 0.994 || y < 0.006 || y > 0.994) off[i] = true;
  }
  // Straight stretches: runs of segments that keep the heading of their first one within 4 deg,
  // measured across the picture (a tall straight object edge, like a door frame, is real).
  for (let i = 0; i < n; ) {
    const a0 = Math.atan2(Y(i + 1) - Y(i), X(i + 1) - X(i));
    let j = i + 1;
    while (j < i + n - 1) {
      const a = Math.atan2(Y(j + 1) - Y(j), X(j + 1) - X(j));
      if (Math.abs(Math.atan2(Math.sin(a - a0), Math.cos(a - a0))) > 0.07) break;
      j++;
    }
    if (Math.abs(X(j) - X(i)) > LONG_CUT * w) for (let k = i; k <= j; k++) off[k % n] = true;
    i = j;
  }
  // Strength: distance along the outline to the nearest point that gets none, eased in.
  const seg = (i: number) => Math.hypot(X(i + 1) - X(i), Y(i + 1) - Y(i));
  const dist = new Array<number>(n).fill(off.some(Boolean) ? Infinity : RIM_FADE * fadeOf);
  for (let pass = 0; pass < 2; pass++)
    for (let k = 0; k < 2 * n; k++) {
      const i = pass === 0 ? k % n : (2 * n - 1 - k) % n;
      if (off[i]) dist[i] = 0;
      else if (pass === 0) dist[i] = Math.min(dist[i], dist[(i + n - 1) % n] + seg(i - 1 + n));
      else dist[i] = Math.min(dist[i], dist[(i + 1) % n] + seg(i));
    }
  const level = dist.map((d) => Math.round(Math.min(1, d / (RIM_FADE * fadeOf)) * (RIM_LEVELS - 1)) / (RIM_LEVELS - 1));
  // Group segments i..i+1 by the lower strength of their ends; a closed rim of one level is one run.
  const runs: { a: number; idx: number[] }[] = [];
  const segA = (i: number) => Math.min(level[i], level[(i + 1) % n]);
  let s0 = 0;
  while (s0 < n && segA((s0 + n - 1) % n) === segA(s0)) s0++;
  if (s0 === n) return segA(0) > 0 ? [{ a: segA(0), idx: [...Array(n + 1).keys()].map((k) => k % n) }] : [];
  let cur: { a: number; idx: number[] } | null = null;
  for (let k = 0; k < n; k++) {
    const i = (s0 + k) % n;
    const a = segA(i);
    if (!cur || cur.a !== a) {
      cur = { a, idx: [i] };
      runs.push(cur);
    }
    cur.idx.push((i + 1) % n);
  }
  return runs.filter((r) => r.a > 0);
}

/** One rim run as a smooth open path through the midpoints, like smoothPath. */
function runPath(ctx: CanvasRenderingContext2D, p: number[], idx: number[], ox: number, oy: number, w: number, h: number) {
  const n = p.length / 2;
  const X = (i: number) => ox + p[(((i % n) + n) % n) * 2] * w;
  const Y = (i: number) => oy + p[(((i % n) + n) % n) * 2 + 1] * h;
  ctx.beginPath();
  ctx.moveTo((X(idx[0]) + X(idx[0] + 1)) / 2, (Y(idx[0]) + Y(idx[0] + 1)) / 2);
  for (let k = 1; k < idx.length; k++) {
    const i = idx[k];
    ctx.quadraticCurveTo(X(i), Y(i), (X(i) + X(i + 1)) / 2, (Y(i) + Y(i + 1)) / 2);
  }
}

// `thin`: a narrow inner band and almost no flat tint, for a camera that fills the screen, where
// the regular fade over a close obstacle covers a third of the picture.
function drawHazards(
  ctx: CanvasRenderingContext2D,
  hz: Hazard[],
  ox: number,
  oy: number,
  w: number,
  h: number,
  thin: boolean,
) {
  // The glowing band inside the outline: about 14 px of falloff on a 1080-high picture.
  const fall = (thin ? 0.006 : 0.013) * h;
  const tint = thin ? 0.3 : 1;
  const now = performance.now();
  const shown = hz.map((x) => [x, look(x, now)] as const);
  looks = looks.filter((l) => now - l.seen < 400);
  for (const [x, l] of [...shown].reverse()) {
    if (x.poly.length < 6 || !l.shape) continue;
    const poly = l.shape;
    const [r, g, b] = l.rgb.map(Math.round);
    const red = l.level === 2;
    // The only pulse on screen: red, about 1.2 Hz (2 Hz is the ceiling).
    const beat = red ? 0.75 + 0.25 * Math.sin(now / 130) : 1;
    const runs = rimRuns(poly, w, h);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    // Inner fade: clip to the shape and stroke the rim wide and soft, so it is brightest at the
    // edge. It stays when the flat tint goes (rim-only), so a close wall still glows at its edge.
    ctx.save();
    smoothPath(ctx, poly, ox, oy, w, h);
    ctx.clip();
    const fa = l.alpha * l.fill;
    if (fa > 0.01) {
      ctx.fillStyle = `rgba(${r},${g},${b},${((red ? 0.1 : 0.05) * tint * fa).toFixed(3)})`;
      ctx.fill();
    }
    // Two soft passes, so the band is brightest at the edge and gone about `fall` inside.
    ctx.shadowBlur = fall * ctx.getTransform().a; // shadows ignore the canvas scale
    for (const [width, alpha] of [[2, 0.28], [1, 0.45]] as const) {
      ctx.lineWidth = fall * width;
      for (const run of runs) {
        const a = l.alpha * beat * run.a;
        ctx.shadowColor = `rgba(${r},${g},${b},${(0.9 * a).toFixed(3)})`;
        ctx.strokeStyle = `rgba(${r},${g},${b},${(alpha * a).toFixed(3)})`;
        runPath(ctx, poly, run.idx, ox, oy, w, h);
        ctx.stroke();
      }
    }
    ctx.restore();
    // Crisp contour, on the same runs: never along the picture border or a long straight cut.
    ctx.shadowBlur = 10;
    ctx.lineWidth = red ? 2.6 : 1.8;
    for (const run of runs) {
      const a = l.alpha * run.a;
      ctx.shadowColor = `rgba(${r},${g},${b},${(0.9 * a).toFixed(3)})`;
      ctx.strokeStyle = `rgba(${r},${g},${b},${(0.95 * a).toFixed(3)})`;
      runPath(ctx, poly, run.idx, ox, oy, w, h);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    // Distance tag just above the part that is in the way, eased with the blob. Kept off the top
    // edge (where the header glass sits in the app) and on screen at the sides.
    const text = `${x.dist.toFixed(1)} m`;
    ctx.font = '600 13px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    const tw = ctx.measureText(text).width + 26;
    // The picture overhangs the screen, so keep the tag inside what is actually visible.
    const left = Math.max(ox, 0);
    const right = Math.min(ox + w, ctx.canvas.clientWidth);
    const tx = Math.min(right - tw / 2 - 6, Math.max(left + tw / 2 + 6, ox + l.x * w));
    const ty = Math.max(Math.max(oy, 0) + Math.max(26, 0.09 * h), oy + l.y * h - 14);
    ctx.fillStyle = `rgba(255,255,255,${(0.86 * l.alpha).toFixed(3)})`;
    ctx.beginPath();
    ctx.roundRect(tx - tw / 2, ty - 13, tw, 24, 12);
    ctx.fill();
    ctx.strokeStyle = `rgba(${r},${g},${b},${(0.7 * l.alpha).toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(tx - tw / 2 + 11, ty - 1, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = `rgba(29,29,31,${l.alpha.toFixed(3)})`;
    ctx.fillText(text, tx - tw / 2 + 19, ty + 4);
  }
}

/** A remembered thing as seen from the camera: normalised frame position, metres away, and how
 *  much it is lit (`bright` 0..1 while it answers a question, `ring` the pulse phase or null). */
export interface ScreenPin {
  label: string;
  x: number;
  y: number;
  dist: number;
  selected: boolean;
  bright: number;
  ring: number | null;
}

// Pins: a calm memory-coloured halo, the same glow as the blob but still. The label shows when
// the pin is selected or within 2 m.
function drawPins(ctx: CanvasRenderingContext2D, pins: ScreenPin[], ox: number, oy: number, w: number, h: number) {
  ctx.font = '600 13px "Atkinson Hyperlegible Next", system-ui, sans-serif';
  for (const p of pins) {
    const x = ox + p.x * w;
    const y = oy + p.y * h;
    const rad = 16 + 6 * p.bright;
    const glow = ctx.createRadialGradient(x, y, 0, x, y, rad);
    glow.addColorStop(0, rgba(MEMORY, 0.55 + 0.35 * p.bright));
    glow.addColorStop(0.35, rgba(MEMORY, 0.22 + 0.2 * p.bright));
    glow.addColorStop(1, rgba(MEMORY, 0));
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = rgba([255, 255, 255], 0.95);
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
    if (p.ring !== null) {
      ctx.strokeStyle = rgba(MEMORY, 0.8 * (1 - p.ring));
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 10 + 34 * p.ring, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (!p.selected && p.dist >= 2) continue;
    const text = `${p.label} · ${p.dist.toFixed(1)} m`;
    const tw = ctx.measureText(text).width + 20;
    const ty = y - rad - 16;
    ctx.fillStyle = "rgba(255,255,255,0.86)";
    ctx.beginPath();
    ctx.roundRect(x - tw / 2, ty - 12, tw, 24, 12);
    ctx.fill();
    ctx.strokeStyle = rgba(MEMORY, 0.6);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#1d1d1f";
    ctx.fillText(text, x - tw / 2 + 10, ty + 4.5);
  }
}

/** A soft arc on the edge of the picture toward a bearing (degrees, clockwise from straight
 *  ahead): wider as the turn gets bigger, gone when you face it. The guide and the people arcs
 *  are this shape. */
export class EdgeArc {
  private bearing = 0;
  private alpha = 0;
  private want: number | null = null;
  private t = 0;
  /** Vertical centre as a share of the picture height, eased; arcs that share an edge stack. */
  private at = 0.5;
  constructor(readonly rgb: RGB) {}
  set(bearingDeg: number | null) {
    this.want = bearingDeg === null ? null : ((((bearingDeg + 180) % 360) + 360) % 360) - 180;
  }
  /** The edge it is heading for: 1 right, -1 left, 0 none. */
  get side(): number {
    return this.want === null || Math.abs(this.want) < 12 ? 0 : Math.sign(this.want);
  }
  draw(ctx: CanvasRenderingContext2D, x0: number, y0: number, w: number, h: number, now: number, at = 0.5) {
    const k = ease(now - this.t);
    this.t = now;
    this.at += (at - this.at) * k;
    const faced = this.want === null || Math.abs(this.want) < 12;
    this.alpha += ((faced ? 0 : 1) - this.alpha) * k;
    if (this.want !== null) this.bearing += (this.want - this.bearing) * k;
    if (this.alpha < 0.01) return;
    const turn = Math.min(1, Math.abs(this.bearing) / 150);
    const right = this.bearing > 0;
    const cx = right ? x0 + w : x0;
    const cy = y0 + h * this.at;
    const rx = Math.min(w * 0.22, 46 + 80 * turn);
    const ry = h * (0.14 + 0.3 * turn) * (1 - 2 * Math.min(0.25, Math.abs(this.at - 0.5)));
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    ctx.clip();
    ctx.translate(cx, cy);
    ctx.scale(rx, ry);
    const glow = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    glow.addColorStop(0, rgba(this.rgb, 0.5 * this.alpha));
    glow.addColorStop(0.55, rgba(this.rgb, 0.3 * this.alpha));
    glow.addColorStop(0.8, rgba(this.rgb, 0.1 * this.alpha));
    glow.addColorStop(1, rgba(this.rgb, 0));
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(0, 0, 1, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    // A soft rim along the inside of the glow, so it reads as an arc; blurred, never a hard line.
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    ctx.clip();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * 0.62, ry * 0.9, 0, 0, Math.PI * 2);
    ctx.shadowColor = rgba(this.rgb, 0.9 * this.alpha);
    ctx.shadowBlur = 18;
    ctx.strokeStyle = rgba(this.rgb, 0.4 * this.alpha);
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.restore();
  }
}

/** The edge arcs over the camera: one toward the guide target, one per person or voice. People
 *  are drawn only when someone reports them (`setPeople`); never a box, a face mark or a name. */
export class EdgeArcs {
  readonly guide = new EdgeArc(GUIDE);
  private people: EdgeArc[] = [];
  setGuide(bearingDeg: number | null) {
    this.guide.set(bearingDeg);
  }
  setPeople(list: { bearingDeg: number }[]) {
    while (this.people.length < list.length) this.people.push(new EdgeArc(PERSON));
    this.people.forEach((a, i) => a.set(list[i]?.bearingDeg ?? null));
  }
  draw(ctx: CanvasRenderingContext2D, x0: number, y0: number, w: number, h: number) {
    const now = performance.now();
    // Arcs on the same edge stack instead of nesting: the guide on top, people below it in turn.
    const g = this.guide.side;
    const count = (side: number) => (g === side ? 1 : 0) + this.people.filter((p) => p.side === side).length;
    const centre = (slot: number, total: number) => (total > 1 ? 0.5 + (slot - (total - 1) / 2) * (0.6 / total) : 0.5);
    const used: Record<number, number> = { [-1]: g === -1 ? 1 : 0, 1: g === 1 ? 1 : 0 };
    for (const a of this.people) a.draw(ctx, x0, y0, w, h, now, a.side ? centre(used[a.side]++, count(a.side)) : 0.5);
    this.guide.draw(ctx, x0, y0, w, h, now, g ? centre(0, count(g)) : 0.5);
  }
}

/** The in-path obstacles over the live video, with the remembered things in view and the edge
 *  arcs when there are any. */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  vw: number,
  vh: number,
  s: Safety | null,
  thin = false,
  extra?: { pins?: ScreenPin[]; arcs?: EdgeArcs },
) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!vw) {
    extra?.arcs?.draw(ctx, 0, 0, cw, ch);
    return;
  }
  // The picture is object-fit: cover, filling the screen; find where it sits, overhanging the canvas.
  const scale = Math.max(cw / vw, ch / vh);
  const w = vw * scale;
  const h = vh * scale;
  extra?.arcs?.draw(ctx, (cw - w) / 2, (ch - h) / 2, w, h);
  if (extra?.pins) {
    // Halos stay on the picture, like the edge arcs; a pin just outside the frame shows no glow.
    ctx.save();
    ctx.beginPath();
    ctx.rect((cw - w) / 2, (ch - h) / 2, w, h);
    ctx.clip();
    drawPins(ctx, extra.pins, (cw - w) / 2, (ch - h) / 2, w, h);
    ctx.restore();
  }
  if (!s || !s.hazards) return;
  drawHazards(ctx, s.hazards, (cw - w) / 2, (ch - h) / 2, w, h, thin);
}

export const MODE_RGB: Record<string, [number, number, number]> = {
  slow: CAUTION,
  fast: DANGER,
  solid: DANGER,
  lost: [0, 82, 168],
  guide: GUIDE,
  find: GUIDE,
  talk: PERSON,
};

/** The belt seen from above, front at the top: the wearer, the strap, and the eight numbered
 *  motors. Each motor is drawn from the belt's own telemetry, so it lights and shakes exactly
 *  when the firmware is driving it. `guideBearing` (degrees, clockwise from the front) overrides
 *  the server's guide bearing when the client has its own target. */
export function drawRing(canvas: HTMLCanvasElement, b: Belt | null, s: Safety | null, guideBearing?: number | null) {
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  if (size < 20) return;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const c = size / 2;
  const R = c - 24;
  const now = performance.now();
  // Wearer from above: shoulders and head, nose pointing forward.
  ctx.fillStyle = "rgba(0,102,204,0.07)";
  ctx.beginPath();
  ctx.ellipse(c, c, R * 0.62, R * 0.4, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(0,102,204,0.16)";
  ctx.beginPath();
  ctx.arc(c, c, R * 0.22, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(c - 5, c - R * 0.2);
  ctx.lineTo(c, c - R * 0.32);
  ctx.lineTo(c + 5, c - R * 0.2);
  ctx.fill();
  // The strap.
  ctx.strokeStyle = "rgba(0,102,204,0.2)";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(c, c, R, 0, Math.PI * 2);
  ctx.stroke();
  // Guiding to a remembered item: the strap glows toward it, wider as the turn gets bigger, the
  // same rule as the arc on the screen edge.
  const gb = guideBearing ?? s?.guide?.bearing ?? null;
  if (gb !== null) {
    const a = ((gb - 90) * Math.PI) / 180;
    const half = ((10 + 40 * Math.min(1, Math.abs(gb) / 150)) * Math.PI) / 180;
    ctx.lineCap = "round";
    ctx.shadowColor = rgba(GUIDE, 0.9);
    ctx.shadowBlur = 12;
    ctx.strokeStyle = rgba(GUIDE, 0.85);
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(c, c, R, a - half, a + half);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.lineCap = "butt";
  }
  ctx.fillStyle = "#0052a8";
  ctx.font = '600 9px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
  ctx.textAlign = "center";
  ctx.fillText(b?.lost ? "BELT LINK LOST" : b?.found ? "FOUND IT" : "BELT", c, c + R * 0.62);
  ctx.font = '600 10px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < 8; i++) {
    const a = (i * 45 - 90) * (Math.PI / 180);
    const mode = !b ? "off" : b.lost ? "lost" : b.modes[i];
    const duty = b ? b.duty[i] / 255 : 0;
    const rgb = MODE_RGB[mode];
    // A running motor shakes by a pixel or two, so the vibration is visible in a still frame too.
    const shake = duty * 1.8;
    const x = c + R * Math.cos(a) + shake * Math.sin(now / 9 + i);
    const y = c + R * Math.sin(a) + shake * Math.cos(now / 11 + i);
    if (rgb && duty > 0) {
      const [r, g, bl] = rgb;
      for (const k of [1, 2]) {
        ctx.strokeStyle = `rgba(${r},${g},${bl},${(0.5 * duty / k).toFixed(3)})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 11 + 6 * k * duty, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.shadowColor = `rgba(${r},${g},${bl},0.9)`;
      ctx.shadowBlur = 14 * duty;
    }
    const lit = rgb ? `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.35 + 0.65 * duty).toFixed(3)})` : "rgba(240,247,255,0.95)";
    ctx.fillStyle = lit;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = mode === "unknown" ? "rgba(0,102,204,0.35)" : rgb ? "rgba(255,255,255,0.9)" : "rgba(0,102,204,0.3)";
    ctx.setLineDash(mode === "unknown" ? [2, 2] : []);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = rgb && duty > 0 ? "#fff" : "#0052a8";
    ctx.fillText(String(i + 1), x, y + 0.5);
  }
}

/** Eight cells under the video: metres and a bar per direction. */
export class BeltCells {
  private cells: HTMLElement[] = [];
  constructor(private root: HTMLElement) {
    for (const name of SECTOR_NAMES) {
      const el = document.createElement("div");
      el.className = "cell";
      el.innerHTML = `<span><em>${SECTOR_NAMES.indexOf(name) + 1}</em>${name.replace("-right", "-R").replace("-left", "-L")}</span><b>-</b><div class="bar"><i></i></div><em class="motor"></em>`;
      root.appendChild(el);
      this.cells.push(el);
    }
  }
  /** Distances, from the safety loop. */
  update(s: Safety): void {
    s.sectors.forEach((sd, i) => {
      const el = this.cells[i];
      // The front motor answers to the walking path too, so show the same number it acts on.
      const d = i === 0 && s.path !== null ? Math.min(sd ?? Infinity, s.path) : sd;
      el.querySelector("b")!.textContent = !s.known[i] ? "no data" : d === null ? "clear" : `${d.toFixed(1)} m`;
      const fill = d === null || !s.known[i] ? 0 : Math.max(0.04, Math.min(1, 1 - d / 3));
      (el.querySelector("i") as HTMLElement).style.width = `${fill * 100}%`;
    });
  }
  /** What each motor is doing right now, from the belt's own telemetry. */
  motors(b: Belt): void {
    this.root.classList.toggle("lost", b.lost);
    b.duty.forEach((v, i) => {
      const el = this.cells[i];
      const mode = b.lost ? "lost" : b.modes[i];
      el.className = `cell ${mode === "off" ? "" : mode}`;
      el.style.setProperty("--duty", (v / 255).toFixed(2));
    });
  }
}

/** The belt's own telemetry as 3D cells: each motor as bright as the firmware drives it. */
export function cellsFromBelt(b: Belt): Cell[] {
  return b.duty.map((d, i) => {
    const rgb = MODE_RGB[b.lost ? "lost" : b.modes[i]] ?? null;
    return { rgb, level: rgb ? d / 255 : 0 };
  });
}

/** With no belt connected, the server's cues as 3D cells, pulsing at each cue's own rate. */
export function cellsFromCues(s: Safety | null, nowMs: number): Cell[] {
  return Array.from({ length: 8 }, (_, i) => {
    const q = s?.cues[i];
    const rgb = q ? (MODE_RGB[q.mode] ?? null) : null;
    if (!q || !rgb) return { rgb: null, level: 0 };
    const pulse = q.rate_hz > 0 ? 0.55 + 0.45 * Math.abs(Math.sin((nowMs / 1000) * Math.PI * q.rate_hz)) : 1;
    return { rgb, level: Math.max(0.35, Math.min(1, q.intensity)) * pulse };
  });
}
