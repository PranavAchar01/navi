// The judge view (/judge.html, or /?view=judge): one 16:9 screen showing a walk end to end.
// The glasses camera is the whole screen. Over it, on glass, only what the wearer feels: the
// belt's eight buzzers all the time, and a short caption when they ask and the house answers,
// with the same answer spelled in Braille while the wrist band plays it.

import "./judge.css";
import { Net } from "./net";
import { drawOverlay } from "./overlay";
import {
  STOP,
  WarningFilter,
  ago,
  cueWarning,
  distanceText,
  roundDistance,
  type Warning,
} from "./lens";
import type { Activity, PinJson, Safety } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const net = new Net();
const frameBox = $("frame");
const cam = $<HTMLCanvasElement>("cam");
const camOverlay = $<HTMLCanvasElement>("cam-overlay");
const beltCanvas = $<HTMLCanvasElement>("belt");

let latest: Safety | null = null;
const filter = new WarningFilter();
// The warning as of the last 100 ms refresh (the phone's lens rate). The belt and the camera tag
// both print this one number, so they never disagree by a rounding step or a frame.
let said: Warning | null = null;
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
  source.className = `j-source glass ${replay ? "replay" : "live"}`;
});
// The wearer's activity changes what the belt and the speaker do, so the judges see it. The
// server starts in Guide; a viewer who joins later gets the last switch replayed.
const ACTIVITY_NAMES: Record<Activity, string> = {
  guide: "Guide",
  find: "Find",
  talk: "Talk",
  explore: "Explore",
};
const activity = $("activity");
net.on("activity", (a) => {
  if (activity.dataset.mode === a.mode) return;
  activity.dataset.mode = a.mode;
  activity.querySelector("span")!.textContent = ACTIVITY_NAMES[a.mode];
  activity.classList.remove("changed");
  void activity.offsetWidth; // restart the flash
  activity.classList.add("changed");
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

// Cover-fit with the picture's own aspect, so the overlay's contain-fit lands on it exactly. A
// portrait stream on a wide screen shows a band of the picture: it rides down to the nearest
// obstacle's tag when there is one, and otherwise sits a little below the middle, on the floor
// ahead, where the walking path is.
let focus = 0.55;
function placeFrame(s: Safety | null) {
  if (!cam.width) return;
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const k = Math.max(sw / cam.width, sh / cam.height);
  const w = cam.width * k;
  const h = cam.height * k;
  let want = 0.55;
  const near = s?.hazards?.length ? s.hazards.reduce((a, b) => (b.dist < a.dist ? b : a)) : null;
  if (near && h > sh) want = Math.min(1, Math.max(0, (near.tag[1] * h - sh * 0.62) / (h - sh)));
  focus += (want - focus) * 0.03;
  // Size only when it changes (a layout); the slide is a transform, which the compositor moves
  // without laying the page out again every frame.
  const size = `${w}x${h}`;
  if (frameBox.dataset.size !== size) {
    frameBox.dataset.size = size;
    frameBox.style.width = `${w}px`;
    frameBox.style.height = `${h}px`;
  }
  frameBox.style.transform = `translate3d(${(sw - w) / 2}px, ${-(h - sh) * focus}px, 0)`;
}

// The caption: the question while it is heard, then the answer and its Braille, then gone.
const caption = $("caption");
const question = $("question");
let answerPin: PinJson | null = null;
let answerText = "";
let answerLive = false;
let hideAt = 0;
let playing = false;
function holdCaption(seconds: number) {
  hideAt = Math.max(hideAt, Date.now() + seconds * 1000);
  caption.classList.add("show");
}
net.on("transcript", (t) => {
  if (t.final) modelStart("muse");
  question.textContent = t.text;
  caption.classList.remove("answered", "touch");
  caption.classList.add("asking");
  holdCaption(3);
  if (t.final) reach("ask");
});
net.on("answer", (a) => {
  modelDone("muse", a.language === "muse-spark" && a.ms != null ? seconds(a.ms) : null);
  answerPin = a.pin;
  answerText = a.text;
  answerLive = a.live ?? false;
  reach("ask");
  const thumb = $<HTMLImageElement>("answer-thumb");
  thumb.src = a.pin?.thumb ? `data:image/jpeg;base64,${a.pin.thumb}` : "";
  thumb.style.display = a.pin?.thumb ? "" : "none";
  $("answer-label").textContent = a.label ?? a.text;
  $("answer-seen").textContent = seenLine();
  $("answer-by").textContent = byLine(a.live ? undefined : a.pin?.source, a.language);
  caption.classList.remove("asking");
  caption.classList.add("answered");
  holdCaption(3);
  if (a.pin) reach("find");
});

function seenLine(): string {
  if (answerLive) return "in view now";
  if (!answerPin) return answerText;
  const near = answerPin.near[0] ? `by the ${answerPin.near[0]}` : "";
  return [near, `seen ${ago(answerPin.ts * 1000, Date.now())}`].filter(Boolean).join(" · ");
}

// The provider names the server reports, as people say them.
const MODEL_NAMES: Record<string, string> = {
  "muse-spark": "Muse Spark",
  "sam-3.1": "SAM 3.1",
  "yoloe-26l-seg": "YOLOE",
  "yoloe + sam-3.1": "YOLOE + SAM 3.1",
};
function byLine(finder: string | undefined, language: string | undefined): string {
  const found = finder && finder !== "tap" ? `found by ${MODEL_NAMES[finder] ?? finder}` : "";
  const said = language && language !== "template" ? `answer by ${MODEL_NAMES[language] ?? language}` : "";
  return [found, said].filter(Boolean).join(" · ");
}

// The same answer by touch: the wrist band's six motors are the six dots of one Braille cell.
// Each dot lights for as long as the server says its motor buzzes; the word being spelled is
// bright in the cue under it. On the belt fallback the dots go to belt motors instead.
const dotEls = [...document.querySelectorAll<HTMLElement>("#cell i")];
const brailleWords = $("braille-words");
const BELT_OF_DOT = [0, 7, 6, 1, 2, 3]; // dot 1 front, 2 front-left, 3 left, 4 front-right, 5 right, 6 back-right
let cueWords: string[] = [];
let device = "";
let onBelt = false;
let brailleDots: number[] = [];
let dotsOff = 0;
function showWord(k: number) {
  brailleWords.innerHTML = "";
  const head = document.createElement("small");
  head.textContent = `Braille · ${device}`;
  brailleWords.append(head);
  cueWords.forEach((w, i) => {
    const el = document.createElement(i === k ? "b" : "span");
    el.textContent = w;
    brailleWords.append(el, " ");
  });
}
function lightDots(dots: number[], ms: number) {
  clearTimeout(dotsOff);
  brailleDots = dots;
  dotEls.forEach((el, i) => el.classList.toggle("on", dots.includes(i + 1)));
  if (ms > 0) dotsOff = window.setTimeout(() => lightDots([], 0), ms);
}
net.on("braille", (m) => {
  if (m.phase === "start") {
    cueWords = m.cues.split(" ");
    device = m.device;
    onBelt = m.device.includes("belt");
    playing = true;
    caption.classList.add("touch");
    showWord(-1);
  } else if (m.phase === "cell") {
    showWord(m.word);
    lightDots(m.dots, m.dur_ms);
  } else {
    lightDots([], 0);
    playing = false;
    holdCaption(3);
  }
});

// The Meta models in the loop, one small pill each with the one number measured on its last run,
// lit while it works: mapping while windows keep landing, SAM 3.1 when it was called in the
// last window, Muse Spark from the end of the question to the answer, DAT while the glasses
// stream.
const modelEls = new Map(
  [...document.querySelectorAll<HTMLElement>("#models li")].map((li) => [li.dataset.model!, li]),
);
const busyUntil = new Map<string, number>();
const seconds = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
function modelStart(name: string, holdMs = 15000) {
  busyUntil.set(name, performance.now() + holdMs);
  modelEls.get(name)!.classList.add("seen");
}
function modelDone(name: string, value: string | null, holdMs = 0) {
  busyUntil.set(name, performance.now() + holdMs);
  const li = modelEls.get(name)!;
  if (value) {
    li.querySelector("b")!.textContent = value;
    li.classList.add("seen");
  }
}
let samImages = 0;
net.on("memory", (m) => {
  // Windows land every few seconds while the walk is mapped; a gap this long means it stopped.
  modelDone("map", seconds(m.infer_ms), 12000);
  if (m.sam_images != null && m.sam_images > samImages && m.sam_ms) {
    modelDone("sam", seconds(m.sam_ms), 12000);
  }
  samImages = m.sam_images ?? samImages;
});

setInterval(() => {
  const now = performance.now();
  const dat = latest?.camera === "glasses" && now - lastFrameAt < 1000 && latest.in_fps;
  if (dat) modelDone("dat", `${Math.round(dat)} fps`, 1000);
  for (const [name, li] of modelEls) li.classList.toggle("on", now < (busyUntil.get(name) ?? 0));
  said = filter.shown && { ...filter.shown };
  if (answerPin) $("answer-seen").textContent = seenLine();
  if (caption.classList.contains("show") && !playing && Date.now() > hideAt) {
    caption.classList.remove("show");
  }
}, 100);

/** The belt from above: the wearer in the middle, eight buzzers glowing amber as hard as the
 *  server drives them, the guide buzzer blue, Braille dots lilac. The one number sits inside. */
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
  const R = size * 0.36;
  const now = performance.now();
  // Shoulders and head, nose forward.
  ctx.fillStyle = "rgba(255,255,255,0.16)";
  ctx.beginPath();
  ctx.ellipse(c, c, R * 0.42, R * 0.26, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.32)";
  ctx.beginPath();
  ctx.arc(c, c, R * 0.15, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(c - R * 0.06, c - R * 0.135);
  ctx.lineTo(c, c - R * 0.24);
  ctx.lineTo(c + R * 0.06, c - R * 0.135);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = size * 0.03;
  ctx.beginPath();
  ctx.arc(c, c, R, 0, Math.PI * 2);
  ctx.stroke();
  const cues = s?.cues ?? [];
  const motor = size * 0.06;
  for (let i = 0; i < 8; i++) {
    const a = ((i * 45 - 90) * Math.PI) / 180;
    const q = cues[i];
    const mode = (q?.mode as string) ?? "off";
    const guide = mode === "guide";
    const touch = onBelt && brailleDots.some((d) => BELT_OF_DOT[d - 1] === i);
    // Only obstacle cues buzz amber; "lost" (no trusted data) and "off" stay dark.
    const buzzing = mode === "slow" || mode === "fast" || mode === "solid";
    const strength = touch ? 1 : buzzing ? Math.max(0.35, Math.min(1, q.intensity)) : 0;
    const on = strength > 0 || guide;
    const rgb = touch ? [201, 166, 255] : guide ? [90, 176, 255] : [255, 190, 70];
    const k = guide ? 0.85 : strength;
    // Fast and solid buzz harder than slow: the glow breathes at the cue's own rate.
    const rate = q?.rate_hz ?? 0;
    const pulse = on && rate > 0 ? 0.7 + 0.3 * Math.abs(Math.sin((now / 1000) * Math.PI * rate)) : 1;
    const x = c + R * Math.cos(a);
    const y = c + R * Math.sin(a);
    if (on) {
      const glow = ctx.createRadialGradient(x, y, motor * 0.6, x, y, motor * (1.6 + 1.6 * k));
      glow.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.55 * k * pulse).toFixed(3)})`);
      glow.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, motor * (1.6 + 1.6 * k), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = on
      ? `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.45 + 0.55 * k * pulse).toFixed(3)})`
      : "rgba(255,255,255,0.14)";
    ctx.beginPath();
    ctx.arc(x, y, motor, 0, Math.PI * 2);
    ctx.fill();
  }
  // One number inside the ring, under the wearer: the obstacle when there is one, otherwise the
  // distance to what they are being guided to.
  const w = said;
  const g = s?.guide;
  let text: string | null = null;
  let color = "#fff";
  if (w && w.band > 0) text = w.band === STOP ? "Stop" : distanceText(w.dist);
  else if (g && g.dist !== null) {
    text = distanceText(g.dist);
    color = "rgb(90,176,255)";
  }
  if (text) {
    ctx.font = `800 ${Math.round(size * 0.1)}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.fillText(text, c, c + R * 0.56);
  }
}

/** The frame's hazards with their tags in the same rounding, and the nearest one carrying the
 *  exact number the belt shows. */
function agreeing(s: Safety | null): Safety | null {
  if (!s?.hazards?.length) return s;
  const hazards = s.hazards.map((h) => ({ ...h, dist: roundDistance(h.dist) }));
  if (said && said.dist !== null) {
    const near = hazards.reduce((a, b) => (b.dist < a.dist ? b : a));
    near.dist = roundDistance(said.dist);
  }
  return { ...s, hazards };
}

function frame() {
  // The overlay belongs to the frame on screen, not the newest message.
  const s = bySeq.get(viewSeq) ?? latest;
  const fresh = performance.now() - lastFrameAt < 2000;
  placeFrame(fresh ? s : null);
  drawOverlay(camOverlay, cam.width, cam.height, fresh ? agreeing(s) : null, true);
  drawBelt(latest);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
net.connect();
