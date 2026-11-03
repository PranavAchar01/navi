// Demo capture ("the port"): /capture.html?capture=<scene>, or /?capture=<scene>, which redirects
// here. A clean stage with no app chrome, 1920x1080 (1080x1920 with &vertical=1), for recording
// with real-time tab capture (scripts/capture.mjs). The feed is whatever the server forwards: the
// glasses live, or a recorded walk replayed at its own timing by scripts/judge_replay.py.
//
// Scenes: obstacle, recall, navigate, map, belt, split.
// &ask=keys asks "where are my keys" (or &ask=<a whole question>) &ask-at=6 seconds after the
// first frame. &period=48 is the map orbit's seconds per turn. &label-radius=2 (map units).
// The answer card stays up until the take ends; &hold-answer=0 hides it after 12 s like the app.
// The renderers are the app's own (overlay.ts, map3d.ts); this file only lays them out.

import "./capture.css";
import { ago, distanceText } from "./lens";
import { MapView } from "./map3d";
import { Net } from "./net";
import { drawOverlay, drawRing } from "./overlay";
import type { Answer, Belt, PinJson, Safety } from "./types";

type Scene = "obstacle" | "recall" | "navigate" | "map" | "belt" | "split";
const SCENES: Scene[] = ["obstacle", "recall", "navigate", "map", "belt", "split"];

const q = new URLSearchParams(location.search);
const scene: Scene = SCENES.includes(q.get("capture") as Scene) ? (q.get("capture") as Scene) : "obstacle";
const vertical = q.get("vertical") === "1";
const W = vertical ? 1080 : 1920;
const H = vertical ? 1920 : 1080;
const askParam = q.get("ask");
const askText = askParam ? (askParam.includes(" ") ? askParam : `where are my ${askParam}`) : "";
const askAt = Number(q.get("ask-at") ?? 6);
const period = Number(q.get("period") ?? 48);
const holdAnswer = q.get("hold-answer") !== "0";

// What scripts/capture.mjs waits on before it starts recording.
const state = { scene, frames: 0, points: 0, answers: 0, asked: false, frameGap: 0 }; // frameGap: longest wait between shown camera frames, ms (the recorder zeroes it)
(window as unknown as { __capture: typeof state }).__capture = state;

const stage = document.getElementById("stage")!;
stage.classList.add(`s-${scene}`);
stage.classList.toggle("vertical", vertical);
stage.style.width = `${W}px`;
stage.style.height = `${H}px`;
// Fit the stage to a preview window; the recorder opens it at exactly W x H, so scale 1.
function fitStage() {
  const k = Math.min(innerWidth / W, innerHeight / H);
  stage.style.transform = k < 0.999 || k > 1.001 ? `scale(${k})` : "";
}
fitStage();
addEventListener("resize", fitStage);

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, parent: HTMLElement = stage) {
  const e = document.createElement(tag);
  e.className = cls;
  parent.appendChild(e);
  return e;
}

// ---------------------------------------------------------------- camera
/** The camera picture, contained (the glasses frame is 9:16), over a blurred cover of itself so
 *  a landscape stage has no black bars. The hazard overlay sits on the same box. */
class Cam {
  readonly box: HTMLDivElement;
  private back: HTMLCanvasElement;
  private pic: HTMLCanvasElement;
  private ov: HTMLCanvasElement;
  vw = 0;
  vh = 0;
  constructor(cls: string) {
    this.box = el("div", `cap-cam ${cls}`);
    this.back = el("canvas", "cap-back", this.box);
    this.pic = el("canvas", "cap-pic", this.box);
    this.ov = el("canvas", "cap-ov", this.box);
  }
  show(bmp: ImageBitmap) {
    const fresh = bmp.width !== this.vw || bmp.height !== this.vh;
    this.vw = bmp.width;
    this.vh = bmp.height;
    const dpr = window.devicePixelRatio || 1;
    const cw = this.box.clientWidth;
    const ch = this.box.clientHeight;
    if (!cw || !ch) return;
    // The picture and the overlay canvases cover only the contained picture, not the whole box:
    // a full-stage 4K overlay redrawn every paint was what stalled the compositor.
    const k = Math.min(cw / bmp.width, ch / bmp.height);
    const w = Math.round(bmp.width * k);
    const h = Math.round(bmp.height * k);
    if (fresh || this.pic.clientWidth !== w || this.pic.clientHeight !== h) {
      for (const c of [this.pic, this.ov])
        Object.assign(c.style, { left: `${(cw - w) / 2}px`, top: `${(ch - h) / 2}px`, width: `${w}px`, height: `${h}px` });
      this.pic.width = Math.round(w * dpr);
      this.pic.height = Math.round(h * dpr);
    }
    const ctx = this.pic.getContext("2d")!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, this.pic.width, this.pic.height);
    // The backdrop: a tiny cover-crop in the box's shape, blurred and dimmed here, at 96 px, and
    // stretched by CSS: far cheaper than a CSS blur on the full-size layer.
    const bw = Math.max(16, Math.round((96 * cw) / Math.max(cw, ch)));
    const bh = Math.max(16, Math.round((96 * ch) / Math.max(cw, ch)));
    if (this.back.width !== bw || this.back.height !== bh) {
      this.back.width = bw;
      this.back.height = bh;
    }
    const kb = Math.max(bw / bmp.width, bh / bmp.height);
    const bctx = this.back.getContext("2d")!;
    bctx.filter = "blur(2.5px) brightness(0.55) saturate(1.3)";
    bctx.drawImage(bmp, (bw - bmp.width * kb) / 2, (bh - bmp.height * kb) / 2, bmp.width * kb, bmp.height * kb);
  }
  overlay(s: Safety | null) {
    drawOverlay(this.ov, this.vw, this.vh, s);
  }
}

// ---------------------------------------------------------------- belt ring
/** drawRing sizes its motors and labels in CSS pixels for a ~170 px ring, so a big ring is drawn
 *  at that size with a raised pixel ratio and scaled up by CSS: crisp, and the same drawing. */
class Ring {
  private canvas: HTMLCanvasElement;
  constructor(cls: string, base: number, private scale: number) {
    const wrap = el("div", `cap-ring ${cls}`);
    wrap.style.width = wrap.style.height = `${base * scale}px`;
    this.canvas = el("canvas", "", wrap);
    this.canvas.style.width = this.canvas.style.height = `${base}px`;
    this.canvas.style.transform = `scale(${scale})`;
  }
  draw(b: Belt | null, s: Safety | null) {
    const own = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");
    const dpr = window.devicePixelRatio || 1;
    Object.defineProperty(window, "devicePixelRatio", { configurable: true, get: () => dpr * this.scale });
    try {
      drawRing(this.canvas, b, s);
    } finally {
      if (own) Object.defineProperty(window, "devicePixelRatio", own);
      else delete (window as { devicePixelRatio?: number }).devicePixelRatio;
    }
  }
}

// ---------------------------------------------------------------- map
/** A sample of the map's points, for the box the camera frames. */
class Extent {
  private xs: number[] = [];
  private ys: number[] = [];
  private zs: number[] = [];
  clear() {
    this.xs = [];
    this.ys = [];
    this.zs = [];
  }
  add(n: number, pts: Float32Array) {
    const step = Math.max(1, Math.floor(n / 4000));
    for (let i = 0; i < n; i += step) {
      this.xs.push(pts[i * 3]);
      this.ys.push(pts[i * 3 + 1]);
      this.zs.push(pts[i * 3 + 2]);
    }
  }
  box(): { c: [number, number, number]; r: number; floor: number } | null {
    if (this.xs.length < 50) return null;
    const pct = (a: number[], f: number) => {
      const s = [...a].sort((x, y) => x - y);
      return s[Math.floor(f * (s.length - 1))];
    };
    // The centre from the 10th to 90th percentile, the size from the 5th to 95th: the dense part
    // of the house, not the stray points out of the windows.
    const c: [number, number, number] = [
      (pct(this.xs, 0.1) + pct(this.xs, 0.9)) / 2,
      (pct(this.ys, 0.1) + pct(this.ys, 0.9)) / 2,
      (pct(this.zs, 0.1) + pct(this.zs, 0.9)) / 2,
    ];
    const r = Math.max(1.5, Math.hypot(pct(this.xs, 0.95) - pct(this.xs, 0.05), pct(this.zs, 0.95) - pct(this.zs, 0.05)) / 2);
    return { c, r, floor: pct(this.ys, 0.03) };
  }
}

const ease = (k: number) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);

/** A slow orbit of the whole map: eases in from wide, then turns at a constant rate while the
 *  height breathes a little. Every term is continuous, so there is never a jump. */
function orbit(map: MapView, ext: Extent, t: number) {
  const b = ext.box();
  if (!b) return;
  const k = ease(Math.min(1, t / 7));
  const theta = 0.6 + (2 * Math.PI * t) / period;
  const r = b.r * (2.3 - 0.65 * k);
  const h = b.r * (1.5 - 0.45 * k) + 0.15 * b.r * Math.sin((2 * Math.PI * t) / (period * 0.6));
  const [cx, , cz] = b.c;
  const ty = b.floor + 0.6;
  map.lookFrom([cx + r * Math.cos(theta), ty + h, cz + r * Math.sin(theta)], [cx, ty, cz]);
}

// ---------------------------------------------------------------- layout per scene
const wantsCam = scene !== "map";
const wantsMap = scene === "recall" || scene === "navigate" || scene === "map" || scene === "split";
const cam = wantsCam
  ? new Cam(scene === "navigate" ? "inset" : scene === "split" ? "half" : scene === "belt" ? "side" : "full")
  : null;
let map: MapView | null = null;
const ext = new Extent();
if (wantsMap) {
  const box = el("div", `cap-map ${scene === "recall" ? "inset" : scene === "split" ? "half" : "full"}`);
  map = new MapView(el("canvas", "", box));
}
const ring =
  scene === "obstacle"
    ? new Ring("corner", 200, vertical ? 1.7 : 1.5)
    : scene === "belt"
      ? new Ring("big", 200, vertical ? 3.6 : 3.8)
      : null;
const card = scene === "recall" || scene === "navigate" || scene === "split" ? el("div", "cap-card") : null;
const guide = scene === "navigate" ? el("div", "cap-guide") : null;

// ---------------------------------------------------------------- feed
const net = new Net();
let latest: Safety | null = null;
let belt: Belt | null = null;
let pins: PinJson[] = [];
const bySeq = new Map<number, Safety>();
let viewSeq = -1;
let firstFrameAt = 0;

net.on("safety", (s) => {
  latest = s;
  bySeq.set(s.seq, s);
  bySeq.delete(s.seq - 60);
  map?.setLiveYaw(s.yaw ?? null);
});
net.on("belt", (b) => (belt = b));
// The server sends a frame each time it finishes one, so they come in bursts and stalls. A take
// shows them from a short queue at the feed's average pace instead: about 250 ms later, but
// evenly spaced, with the overlay still drawn from each frame's own safety message.
const queue: { seq: number; bmp: ImageBitmap }[] = [];
let pace = 66; // ms between frames, the feed's running average
let lastFrameAt = 0;
let nextShow = 0;
net.onCameraFrame(async (seq, jpeg) => {
  if (!cam) return;
  queue.push({ seq, bmp: await createImageBitmap(jpeg) });
  const now = performance.now();
  if (lastFrameAt) pace = 0.95 * pace + 0.05 * Math.min(250, now - lastFrameAt);
  lastFrameAt = now;
});
function present(now: number) {
  if (!cam || queue.length < (nextShow ? 1 : 4) || now < nextShow) return;
  if (queue.length > 8) queue.splice(0, queue.length - 4).forEach((f) => f.bmp.close()); // fell behind
  const f = queue.shift()!;
  cam.show(f.bmp);
  f.bmp.close();
  viewSeq = f.seq;
  state.frames++;
  if (firstFrameAt) state.frameGap = Math.max(state.frameGap, now - shownAt);
  else firstFrameAt = now;
  shownAt = now;
  // Hold three or four frames in hand: a deeper queue plays a little faster, a shallow one slower.
  // Never slower than 16 fps, so a 30 fps take never holds one picture for more than two frames.
  const step = Math.min(62, pace * (queue.length > 4 ? 0.9 : queue.length < 3 ? 1.1 : 1));
  nextShow = Math.max(nextShow + step, now + step / 2);
}
let shownAt = 0;

// Frame the whole map once it has some shape, and again each time it doubles or is replaced,
// until an answer flies the camera to its pin.
let fittedAt = 0;
function mapData(n: number, pts: Float32Array) {
  ext.add(n, pts);
  state.points += n;
  if (scene !== "map" && !state.answers && state.points > 2000 && state.points >= 2 * fittedAt) {
    fittedAt = state.points;
    map?.fit();
  }
}
net.onMapChunk((n, p, c, z) => {
  map?.addChunk(n, p, c, z);
  mapData(n, p);
});
net.onMapReplace((n, p, c, z) => {
  map?.replaceMap(n, p, c, z);
  ext.clear();
  state.points = 0;
  fittedAt = 0;
  mapData(n, p);
});
net.on("trail", (m) => map?.setTrail(m.points));
net.on("pose", (m) => map?.setPose(m));
net.on("pins", (m) => {
  pins = m.pins;
  map?.setPins(pins);
});
if (scene === "navigate") map?.setFollow(true);
// Labels only near the wearer (and on the answer pin). A replayed map's scale can be off, so
// &label-radius= tightens it.
if (map) map.labelRadius = Number(q.get("label-radius") ?? 2);

// The answer card: the photo the house kept, the spoken answer, and when it was last seen.
let cardUntil = 0;
net.on("answer", (a: Answer) => {
  state.answers++;
  if (a.pin) map?.flyTo(a.pin);
  if (!card) return;
  const seen = a.pin ? (a.live ? "in view now" : `seen ${ago(a.pin.ts * 1000, Date.now())}`) : "";
  card.innerHTML = `${a.pin?.thumb ? `<img alt="" src="data:image/jpeg;base64,${a.pin.thumb}"/>` : ""}<div><p></p><span></span></div>`;
  card.querySelector("p")!.textContent = a.text;
  card.querySelector("span")!.textContent = seen;
  card.classList.add("on");
  // Held for the rest of the take; &hold-answer=0 keeps the app's 12 s.
  cardUntil = holdAnswer ? Infinity : performance.now() + 12_000;
});

// Navigate: point the map at the pin being guided to, once per item.
let guidedTo = "";
function showGuide(s: Safety | null) {
  if (!guide) return;
  const g = s?.guide;
  guide.classList.toggle("on", Boolean(g));
  if (!g) return;
  guide.textContent = s?.found ? `Found: ${g.label}` : `To the ${g.label} · ${g.dist === null ? "finding the way" : distanceText(g.dist)}`;
  if (g.label !== guidedTo) {
    const pin = pins.filter((p) => p.label === g.label).sort((a, b) => b.ts - a.ts)[0];
    if (pin) {
      guidedTo = g.label;
      map?.setGuideTarget(pin);
      map?.flyTo(pin, false);
    }
  }
}

const t0 = performance.now();
function frame(now: number) {
  present(now);
  const s = bySeq.get(viewSeq) ?? latest;
  cam?.overlay(s);
  ring?.draw(belt, s);
  showGuide(s);
  if (card && cardUntil && now > cardUntil) {
    card.classList.remove("on");
    cardUntil = 0;
  }
  if (scene === "map" && map) orbit(map, ext, (now - t0) / 1000);
  const since = firstFrameAt || (scene === "map" && state.points ? t0 : 0);
  if (askText && !state.asked && since && now - since > askAt * 1000) {
    state.asked = true;
    net.send({ type: "ask", text: askText });
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
net.connect();
