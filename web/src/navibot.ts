// Navi, the HUD's companion: a little glass map pin with two eyes. Its eyes follow the thing it
// is guiding you to, its cheeks warm from yellow to red as the belt buzzes harder, it narrows
// its eyes at an obstacle, looks up while you talk, beams when the thing is found and says a
// short line in a bubble. The mood comes from navimood.ts; every value here is a spring, so it
// morphs from one feeling to the next and never stops moving. One SVG, mutated per frame.

import { heat } from "./beltlayout";
import { type Emote, type Mood, gazeOf } from "./navimood";
import { PRESETS, Spring, clamp, reducedMotion } from "./spring";

interface Pose {
  w: number; // eye width
  h: number; // eye height
  gap: number; // eye centre distance
  ey: number; // eye centre height
  lo: number; // lid at the outer corners, 0..1 of eye height
  li: number; // lid at the inner corners
  happy: number; // eyes as ^ ^
  smile: number; // mouth curve, -1 frown .. 1 grin
  tilt: number; // deg, the pin leaning on its tip
  sx: number;
  sy: number;
  by: number; // lift, negative is up
  dim: number;
}

const BASE: Pose = { w: 10, h: 14, gap: 22, ey: -4, lo: 0, li: 0, happy: 0, smile: 0.45, tilt: 0, sx: 1, sy: 1, by: 0, dim: 0 };

const MOOD: Record<Mood, Partial<Pose>> = {
  idle: {},
  listening: { w: 11, h: 16, sy: 1.04, smile: 0.2 },
  thinking: { h: 12, lo: 0.1, li: 0.3, smile: 0 },
  guiding: { w: 10.5, h: 13, lo: 0.12, li: 0.22, smile: 0.3 },
  careful: { h: 9, lo: 0.25, li: 0.45, smile: -0.2, sy: 0.97 },
  happy: { happy: 1, smile: 1, w: 11 },
  worried: { lo: 0.42, li: 0.02, h: 13, smile: -0.5 },
  sleepy: { h: 5, lo: 0.4, li: 0.4, smile: 0.1, sy: 0.97, by: 1.5 },
  offline: { h: 2.5, dim: 0.55, smile: 0, sy: 0.96, by: 1.5 },
};

const EMOTE: Record<Emote, { pose: Partial<Pose>; ms: number; kick: Partial<Record<keyof Pose, number>> }> = {
  nod: { pose: {}, ms: 300, kick: { by: 70 } },
  hop: { pose: { happy: 0.4 }, ms: 600, kick: { by: -230, sy: 4 } },
  wince: { pose: { h: 4, lo: 0.5, li: 0.5, smile: -0.4 }, ms: 520, kick: { sy: -4 } },
  surprise: { pose: { w: 12.5, h: 19, sy: 1.06, smile: -0.1 }, ms: 700, kick: { by: -80 } },
  delight: { pose: { happy: 1, smile: 1 }, ms: 1300, kick: { by: -260, sy: 5 } },
  shrug: { pose: { lo: 0.2, li: 0, smile: -0.15 }, ms: 900, kick: { tilt: 160 } },
};

const SPRING: Record<keyof Pose, typeof PRESETS.smooth> = {
  w: PRESETS.bouncy,
  h: PRESETS.bouncy,
  gap: PRESETS.smooth,
  ey: PRESETS.smooth,
  lo: PRESETS.smooth,
  li: PRESETS.smooth,
  happy: PRESETS.smooth,
  smile: PRESETS.smooth,
  tilt: PRESETS.wobbly,
  sx: PRESETS.wobbly,
  sy: PRESETS.wobbly,
  by: PRESETS.bouncy,
  dim: PRESETS.gentle,
};

const NS = "http://www.w3.org/2000/svg";
const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
};

/** An eye: an ellipse with its top cut flat by a slanted lid, as one path. side -1 left, 1 right. */
function eyePath(cx: number, cy: number, w: number, h: number, lo: number, li: number, side: number) {
  const pts: string[] = [];
  for (let k = 0; k < 28; k++) {
    const a = (k / 28) * Math.PI * 2;
    const x = Math.cos(a) * (w / 2);
    let y = Math.sin(a) * (h / 2);
    // Outer corner is away from the face's middle.
    const u = (x * side) / w + 0.5; // 0 at the inner corner, 1 at the outer
    const lid = -h / 2 + h * (li + (lo - li) * u);
    if (y < lid) y = lid;
    pts.push(`${(cx + x).toFixed(2)},${(cy + y).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
}

export class NaviBot {
  private springs = {} as Record<keyof Pose, Spring>;
  private gx = new Spring(0, PRESETS.smooth);
  private gy = new Spring(0, PRESETS.smooth);
  private blink = new Spring(0, PRESETS.blink);
  private nextBlink = 0;
  private mood: Mood = "idle";
  private bearing: number | null = null;
  private buzz = 0;
  private hold: { pose: Partial<Pose>; until: number } | null = null;
  private body: SVGGElement;
  private eyes: SVGPathElement[];
  private arcs: SVGPathElement[];
  private glints: SVGCircleElement[];
  private cheeks: SVGEllipseElement[];
  private mouth: SVGPathElement;
  private bubble: HTMLDivElement;
  private bubbleTimer = 0;
  private typeTimer = 0;
  private last = 0;
  private still = reducedMotion();

  constructor(host: HTMLElement) {
    for (const k of Object.keys(BASE) as (keyof Pose)[]) this.springs[k] = new Spring(BASE[k], SPRING[k]);
    const wrap = document.createElement("div");
    wrap.className = "navi";
    const svg = el("svg", { viewBox: "-50 -48 100 112", "aria-hidden": "true" });
    const defs = el("defs", {}, svg);
    const g = el("radialGradient", { id: "navi-glass", cx: "32%", cy: "22%", r: "85%" }, defs);
    el("stop", { offset: "0", "stop-color": "rgb(255 255 255 / 0.92)" }, g);
    el("stop", { offset: "0.45", "stop-color": "rgb(230 241 255 / 0.7)" }, g);
    el("stop", { offset: "1", "stop-color": "rgb(0 102 204 / 0.55)" }, g);
    const rim = el("linearGradient", { id: "navi-rim", x1: "0", y1: "0", x2: "0", y2: "1" }, defs);
    el("stop", { offset: "0", "stop-color": "rgb(255 255 255 / 0.9)" }, rim);
    el("stop", { offset: "1", "stop-color": "rgb(255 255 255 / 0.2)" }, rim);
    // A soft shadow where the tip touches down.
    el("ellipse", { cx: 0, cy: 60, rx: 14, ry: 3, fill: "rgb(0 40 100 / 0.22)", class: "navi-shadow" }, svg);
    this.body = el("g", {}, svg);
    el(
      "path",
      {
        d: "M0 58 C -9 43 -38 27 -38 0 A38 38 0 1 1 38 0 C 38 27 9 43 0 58 Z",
        fill: "url(#navi-glass)",
        stroke: "url(#navi-rim)",
        "stroke-width": 1.4,
      },
      this.body,
    );
    // The glass highlight, top left.
    el("ellipse", { cx: -15, cy: -23, rx: 12, ry: 5.5, fill: "rgb(255 255 255 / 0.55)", transform: "rotate(-32 -15 -23)" }, this.body);
    const face = el("g", {}, this.body);
    this.cheeks = [-1, 1].map((s) => el("ellipse", { cx: s * 18, cy: 8, rx: 6, ry: 3.4 }, face));
    this.eyes = [-1, 1].map(() => el("path", { fill: "#1d1d1f" }, face));
    this.arcs = [-1, 1].map(() =>
      el("path", { fill: "none", stroke: "#1d1d1f", "stroke-width": 3.4, "stroke-linecap": "round" }, face),
    );
    this.glints = [-1, 1].map(() => el("circle", { r: 1.9, fill: "#fff" }, face));
    this.mouth = el("path", { fill: "none", stroke: "#1d1d1f", "stroke-width": 2.4, "stroke-linecap": "round" }, face);
    wrap.appendChild(svg);
    this.bubble = document.createElement("div");
    this.bubble.className = "navi-bubble glass r";
    this.bubble.setAttribute("aria-hidden", "true"); // the app already speaks its answers aloud
    this.bubble.hidden = true;
    wrap.appendChild(this.bubble);
    host.appendChild(wrap);
    requestAnimationFrame((t) => this.frame(t));
  }

  /** The lasting state: mood, the bearing it is guiding to, and how hard the belt buzzes (0..1). */
  set(mood: Mood, bearing: number | null, buzz: number) {
    this.mood = mood;
    this.bearing = bearing;
    this.buzz = clamp(buzz, 0, 1);
  }

  emote(e: Emote) {
    const x = EMOTE[e];
    this.hold = { pose: x.pose, until: performance.now() + x.ms };
    if (this.still) return;
    for (const [k, v] of Object.entries(x.kick) as [keyof Pose, number][]) {
      this.springs[k].impulse(k === "tilt" ? v * (Math.random() < 0.5 ? -1 : 1) : v / (k === "sy" || k === "sx" ? 100 : 1));
    }
  }

  /** A short line in the bubble, typed out, gone after a few seconds. */
  say(text: string, ms = 4200) {
    window.clearInterval(this.typeTimer);
    window.clearTimeout(this.bubbleTimer);
    this.bubble.hidden = false;
    this.bubble.classList.remove("out");
    if (this.still) this.bubble.textContent = text;
    else {
      let n = 0;
      this.bubble.textContent = "";
      this.typeTimer = window.setInterval(() => {
        n += 1;
        this.bubble.textContent = text.slice(0, n);
        // Bounce with the words.
        if (text[n - 1] === " ") this.springs.by.impulse(-18);
        if (n >= text.length) window.clearInterval(this.typeTimer);
      }, 28);
    }
    this.bubbleTimer = window.setTimeout(() => {
      this.bubble.classList.add("out");
      this.bubbleTimer = window.setTimeout(() => (this.bubble.hidden = true), 260);
    }, ms);
  }

  private frame(now: number) {
    requestAnimationFrame((t) => this.frame(t));
    if (document.hidden) return;
    const dt = Math.min(0.05, (now - (this.last || now)) / 1000);
    this.last = now;
    const target: Pose = { ...BASE, ...MOOD[this.mood] };
    if (this.hold && now < this.hold.until) Object.assign(target, this.hold.pose);
    else this.hold = null;
    const gaze = gazeOf(this.mood, this.bearing);
    // The pin leans toward where it is pointing you.
    if (this.mood === "guiding") target.tilt += gaze.x * 9;
    const t = now / 1000;
    if (!this.still) {
      target.sy += Math.sin(t * 2.1) * 0.014; // breathing
      if (this.mood === "careful") target.tilt += Math.sin(t * 30) * 0.8; // a nervous little shiver
      target.by += this.buzz * Math.sin(t * 46) * 0.9; // the belt's buzz comes up through the tip
    }
    for (const k of Object.keys(target) as (keyof Pose)[]) {
      this.springs[k].target = target[k];
      this.springs[k].step(dt);
    }
    this.gx.target = gaze.x;
    this.gy.target = gaze.y;
    this.gx.step(dt);
    this.gy.step(dt);
    if (now > this.nextBlink && this.mood !== "offline") {
      this.blink.target = 1;
      this.nextBlink = now + 2400 + Math.random() * 3200;
    }
    if (this.blink.value > 0.9) this.blink.target = 0;
    this.blink.step(dt);

    const p = Object.fromEntries(Object.entries(this.springs).map(([k, s]) => [k, s.value])) as unknown as Pose;
    this.body.setAttribute(
      "transform",
      `translate(0 ${p.by.toFixed(2)}) rotate(${p.tilt.toFixed(2)} 0 58) translate(0 58) scale(${p.sx.toFixed(3)} ${p.sy.toFixed(3)}) translate(0 -58)`,
    );
    const ox = this.gx.value * 6;
    const oy = this.gy.value * 5;
    const h = Math.max(1.2, p.h * (1 - 0.92 * clamp(this.blink.value, 0, 1)));
    const show = 1 - clamp(p.happy, 0, 1);
    [-1, 1].forEach((s, i) => {
      const cx = (s * p.gap) / 2 + ox;
      const cy = p.ey + oy;
      this.eyes[i].setAttribute("d", eyePath(cx, cy, p.w, h, p.lo, p.li, s));
      this.eyes[i].setAttribute("opacity", (show * (1 - p.dim)).toFixed(3));
      this.arcs[i].setAttribute("d", `M${cx - p.w / 2} ${cy + 2}Q${cx} ${cy - p.h * 0.55} ${cx + p.w / 2} ${cy + 2}`);
      this.arcs[i].setAttribute("opacity", (1 - show).toFixed(3));
      this.glints[i].setAttribute("cx", (cx - p.w * 0.18).toFixed(2));
      this.glints[i].setAttribute("cy", (cy - h * 0.22).toFixed(2));
      this.glints[i].setAttribute("opacity", (show * (h > 5 ? 0.95 : 0)).toFixed(2));
      this.cheeks[i].setAttribute("cx", (s * (p.gap / 2 + 7) + ox * 0.5).toFixed(2));
      this.cheeks[i].setAttribute("cy", (p.ey + 11 + oy * 0.5).toFixed(2));
    });
    // Cheeks: a faint blush at rest, warming from yellow to red with the belt.
    const [r, g, b] = this.buzz > 0.02 ? heat(this.buzz) : [255, 150, 170];
    for (const c of this.cheeks) {
      c.setAttribute("fill", `rgb(${r} ${g} ${b})`);
      c.setAttribute("opacity", (0.32 + 0.55 * this.buzz).toFixed(3));
    }
    const my = p.ey + 15 + oy * 0.6;
    const mw = 4.5 + Math.abs(p.smile) * 1.5;
    this.mouth.setAttribute("d", `M${ox * 0.6 - mw} ${my}Q${ox * 0.6} ${my + p.smile * 6} ${ox * 0.6 + mw} ${my}`);
    this.mouth.setAttribute("opacity", (1 - p.dim).toFixed(3));
  }
}
