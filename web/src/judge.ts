// The judge view (/judge.html, or /?view=judge): one 16:9 screen showing a walk end to end.
// Every panel reads the same server stream, so the camera, the map, the lens, the belt and the
// memory card always describe the same moment.

import "./judge.css";
import { MapView } from "./map3d";
import { Net } from "./net";
import { drawOverlay } from "./overlay";
import {
  BAND_RGB,
  CLEAR,
  STOP,
  LensState,
  WarningFilter,
  ago,
  answerNote,
  cueWarning,
  distanceText,
  drawCard,
  makeCard,
  type AnswerNote,
  type Band,
} from "./lens";
import type { PinJson, Safety } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const net = new Net();
const map = new MapView($<HTMLCanvasElement>("map3d"));
const cam = $<HTMLCanvasElement>("cam");
const camOverlay = $<HTMLCanvasElement>("cam-overlay");
const lensCanvas = $<HTMLCanvasElement>("lens");
const beltCanvas = $<HTMLCanvasElement>("belt");

let latest: Safety | null = null;
let answer: AnswerNote | null = null;
let answerPin: PinJson | null = null;
const filter = new WarningFilter();
const lens = new LensState();
const bySeq = new Map<number, Safety>();
let viewSeq = -1;
let lastFrameAt = 0;

// The story, lit as the stream reaches each beat.
const reached = new Set<string>();
function reach(step: string) {
  if (reached.has(step)) return;
  reached.add(step);
  const items = [...document.querySelectorAll<HTMLElement>("#steps li")];
  let last = -1;
  items.forEach((li, i) => {
    if (reached.has(li.dataset.step!)) last = i;
  });
  items.forEach((li, i) => {
    li.classList.toggle("done", reached.has(li.dataset.step!));
    li.classList.toggle("now", i === last);
  });
}

net.on("safety", (s) => {
  latest = s;
  bySeq.set(s.seq, s);
  bySeq.delete(s.seq - 60);
  const w = filter.update(cueWarning(s), performance.now() / 1000);
  if (w && w.band >= 2) reach("obstacle");
  if (s.guide?.bearing != null) reach("guide");
});

// The source says whether this is the glasses live or a recorded walk played back.
const source = $("source");
net.on("device", (d) => {
  const replay = d.state === "replay";
  source.textContent = replay ? "Simulated walk · recorded footage" : `Live · ${d.name}`;
  source.className = `j-source ${replay ? "replay" : "live"}`;
});
net.onConnection((up) => {
  if (!up) source.textContent = "Reconnecting";
});

net.onCameraFrame(async (seq, jpeg) => {
  const bmp = await createImageBitmap(jpeg);
  if (cam.width !== bmp.width || cam.height !== bmp.height) {
    cam.width = bmp.width;
    cam.height = bmp.height;
  }
  cam.getContext("2d")!.drawImage(bmp, 0, 0);
  bmp.close();
  viewSeq = seq;
  lastFrameAt = performance.now();
  reach("walk");
});

// The house as a floor plan from above: the whole walk fits on screen, one label per kind of
// thing, and the answer is a pulsing ring with a line from the wearer, not a close-up.
map.setTopDown(true);
let refit = 0;
function fitSoon() {
  clearTimeout(refit);
  refit = window.setTimeout(() => map.fit(), 800);
}
net.onMapChunk((n, p, c, z) => {
  map.addChunk(n, p, c, z);
  fitSoon();
});
net.onMapReplace((n, p, c, z) => {
  map.replaceMap(n, p, c, z);
  fitSoon();
});
net.on("trail", (m) => map.setTrail(m.points));
net.on("pose", (m) => map.setPose(m));
let pins: PinJson[] = [];
/** Once there is an answer, its pin is the only one with that label, so the label on the plan
 *  marks the laptop the answer meant and not another one seen earlier. */
function showPins() {
  const a = answerPin;
  map.setPins(a ? [a, ...pins.filter((p) => p.id !== a.id && p.label !== a.label)] : pins);
}
net.on("pins", (m) => {
  pins = m.pins;
  showPins();
});

// What the wearer asked, then what the house memory answered.
const question = $("question");
net.on("transcript", (t) => {
  question.textContent = t.text;
  question.classList.add("show");
  if (t.final) reach("ask");
});
const answerBox = $("answer");
net.on("answer", (a) => {
  answer = answerNote(a, Date.now());
  answerPin = a.pin;
  reach("ask");
  $<HTMLImageElement>("answer-thumb").src = a.pin?.thumb ? `data:image/jpeg;base64,${a.pin.thumb}` : "";
  $<HTMLImageElement>("answer-thumb").style.display = a.pin?.thumb ? "" : "none";
  $("answer-label").textContent = a.label ?? a.text;
  answerBox.classList.add("show");
  if (a.pin) {
    reach("find");
    showPins();
    map.flyTo(a.pin);
  }
});

function seenLine(): string {
  if (!answerPin) return answer?.text ?? "";
  const near = answerPin.near[0] ? `by the ${answerPin.near[0]}` : "";
  return [near, `seen ${ago(answerPin.ts * 1000, Date.now())}`].filter(Boolean).join(" · ");
}

// The lens refreshes every 100 ms, as the phone sends cards to the glasses.
setInterval(() => {
  const card = makeCard(filter.shown, latest?.guide ?? null, answer, Date.now());
  if (card.glyph?.kind === "target") reach("guide");
  drawCard(lensCanvas, lens.refresh(card, Date.now()));
  if (answer) $("answer-seen").textContent = seenLine();
}, 100);

/** The belt from above: the wearer in the middle, eight motors lit by the server's cue for that
 *  direction, in the lens palette. */
function drawBelt(s: Safety | null) {
  const dpr = window.devicePixelRatio || 1;
  const size = beltCanvas.clientWidth;
  if (size < 40) return;
  if (beltCanvas.width !== size * dpr) {
    beltCanvas.width = size * dpr;
    beltCanvas.height = size * dpr;
  }
  const ctx = beltCanvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const c = size / 2;
  const R = size * 0.29;
  const now = performance.now();
  // Shoulders and head, nose forward.
  ctx.fillStyle = "rgba(255,255,255,0.10)";
  ctx.beginPath();
  ctx.ellipse(c, c, R * 0.55, R * 0.34, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  ctx.beginPath();
  ctx.arc(c, c, R * 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(c - R * 0.07, c - R * 0.18);
  ctx.lineTo(c, c - R * 0.3);
  ctx.lineTo(c + R * 0.07, c - R * 0.18);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = size * 0.035;
  ctx.beginPath();
  ctx.arc(c, c, R, 0, Math.PI * 2);
  ctx.stroke();
  const modes = s?.cues?.map((q) => q.mode as string) ?? [];
  const motor = size * 0.065;
  for (let i = 0; i < 8; i++) {
    const a = ((i * 45 - 90) * Math.PI) / 180;
    const mode = modes[i] ?? "off";
    const band = (mode === "slow" ? 1 : mode === "fast" ? 2 : mode === "solid" ? 3 : 0) as Band;
    const guide = mode === "guide";
    const on = band > CLEAR || guide;
    const rgb = guide ? [90, 176, 255] : BAND_RGB[band];
    const rate = band === 3 ? 0 : band === 2 ? 8 : band === 1 ? 3 : 1.5;
    const pulse = rate ? 0.6 + 0.4 * Math.abs(Math.sin((now / 1000) * Math.PI * rate)) : 1;
    const x = c + R * Math.cos(a);
    const y = c + R * Math.sin(a);
    if (on) {
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.22 * pulse).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(x, y, motor * (1.5 + 0.5 * pulse), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = on ? `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.55 + 0.45 * pulse).toFixed(3)})` : "rgba(40,48,70,1)";
    ctx.beginPath();
    ctx.arc(x, y, motor, 0, Math.PI * 2);
    ctx.fill();
  }
  // One number on the belt: the warning the lens shows, on its motor, so the two always agree.
  const w = filter.shown;
  if (w) {
    const a = ((w.sector * 45 - 90) * Math.PI) / 180;
    const rgb = BAND_RGB[w.band];
    ctx.font = `800 ${Math.round(size * 0.07)}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    const text = w.band === STOP ? "Stop" : distanceText(w.dist);
    // Outside the strap, except beside the wearer, where there is no room: there it sits
    // above the motor.
    const side = w.sector === 2 || w.sector === 6;
    const r2 = side ? R : R + motor + size * 0.065;
    const lift = side ? motor + size * 0.05 : 0;
    ctx.fillText(text, c + r2 * Math.cos(a), c + r2 * Math.sin(a) - lift);
  }
}

function frame() {
  // The overlay belongs to the frame on screen, not the newest message.
  const s = bySeq.get(viewSeq) ?? latest;
  drawOverlay(camOverlay, cam.width, cam.height, performance.now() - lastFrameAt < 2000 ? s : null);
  drawBelt(latest);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
net.connect();
