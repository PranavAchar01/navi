import "./styles.css";
import { AudioCues } from "./audio";
import { Source } from "./camera";
import { MapView } from "./map3d";
import { setupGlass } from "./glass";
import { Net } from "./net";
import { BeltCells, drawOverlay, drawRing } from "./overlay";
import { listen, speak } from "./speech";
import type { Answer, Belt, Mode, PinJson, Safety } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const net = new Net();
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const ring = $<HTMLCanvasElement>("ring");
const belt = new BeltCells($("belt"));
const map = new MapView($<HTMLCanvasElement>("map3d"));
// On wide screens the map is the corner minimap: a floor plan, not an orbit view.
const wide = window.matchMedia("(min-width: 901px)");
map.setTopDown(wide.matches);
wide.addEventListener("change", () => map.setTopDown(wide.matches));
const audio = new AudioCues();
const source = new Source(video, net);
let latest: Safety | null = null;
let pins: PinJson[] = [];

// ---------------------------------------------------------------- status chips
const chipState = {
  server: false,
  memory: "loading",
  finder: "",
  fps: 0,
  source: false,
  belt: null as Belt | null,
};
function renderChips() {
  const chip = (cls: string, text: string) => `<span class="chip ${cls}"><i></i>${text}</span>`;
  $("chips").innerHTML = [
    chip(chipState.server ? "ok" : "bad", chipState.server ? "server" : "offline"),
    chip(chipState.memory === "ready" ? "ok" : "", chipState.memory === "ready" ? "memory" : "memory loading"),
    chipState.source ? chip("live", `${chipState.fps.toFixed(0)} fps`) : "",
    chipState.belt
      ? chipState.belt.lost
        ? chip("bad", `${chipState.belt.device}: link lost`)
        : chip("ok", `${chipState.belt.device} · ${chipState.belt.latency_ms?.toFixed(0) ?? "-"} ms`)
      : chip("", "belt offline"),
  ].join("");
}
net.onConnection((up) => {
  chipState.server = up;
  renderChips();
});
net.on("status", (m) => {
  if (m.memory) chipState.memory = m.memory;
  if (m.finder) chipState.finder = m.finder;
  renderChips();
});

// ---------------------------------------------------------------- live view
// Synced display: the video is shown a beat late, frame by frame from a short buffer, so each
// result is drawn on the very frame it was computed from. The delay tracks the measured lag.
const stage = $("stage");
const disp = document.createElement("canvas");
disp.id = "disp";
stage.insertBefore(disp, overlay);
const frames: { t: number; bmp: ImageBitmap }[] = [];
const results: { t: number; s: Safety }[] = [];
let lagMs = 150;
let grabbing = false;
function syncedFrame(now: number): Safety | null {
  if (!video.paused && video.readyState >= 2 && !grabbing) {
    grabbing = true;
    createImageBitmap(video)
      .then((bmp) => frames.push({ t: now, bmp }))
      .finally(() => (grabbing = false));
  }
  const target = now - lagMs - 10;
  let i = -1;
  while (i + 1 < frames.length && frames[i + 1].t <= target) i++;
  if (i >= 0) {
    const f = frames[i];
    if (disp.width !== f.bmp.width) {
      disp.width = f.bmp.width;
      disp.height = f.bmp.height;
    }
    disp.getContext("2d")!.drawImage(f.bmp, 0, 0);
    for (const old of frames.splice(0, i)) old.bmp.close();
  }
  const shownAt = i >= 0 ? frames[0].t : target;
  let r: Safety | null = null;
  for (const x of results) if (x.t <= shownAt + 20) r = x.s;
  while (results.length > 1 && results[1].t <= shownAt) results.shift();
  return r ?? latest;
}

net.on("belt", (m) => {
  const was = chipState.belt?.lost;
  chipState.belt = m;
  belt.motors(m);
  if (was !== m.lost) renderChips();
});
net.on("safety", (m) => {
  latest = m;
  const t = source.sentAt.get(m.seq);
  if (t !== undefined) {
    lagMs = Math.min(600, lagMs * 0.85 + (performance.now() - t) * 0.15);
    results.push({ t, s: m });
  }
  chipState.fps = m.fps;
  belt.update(m);
  audio.update(m);
  const f = (x: number | null, unit: string) => (x === null ? "-" : `${x.toFixed(1)} ${unit}`);
  $("readout").textContent =
    `path ${f(m.path, "m")} · contact ${f(m.ttc, "s")} · low ${f(m.trip, "m")} · drop ${f(m.drop, "m")}` +
    ` · floor ${m.floor ? "ok" : "lost"} · scale ${m.scale.toFixed(2)} · depth ${m.ms.depth?.toFixed(0)} ms` +
    (m.found
      ? " · found it"
      : m.guide
        ? ` · guiding to ${m.guide.label} ${m.guide.dist === null ? "(waiting for the map)" : f(m.guide.dist, "m")}`
        : "");
  renderChips();
});
// Viewer: when another client (the glasses app) is the camera, show the frames the server
// forwards, each with the result computed from it.
let viewing = false;
let viewSeq = -1;
const bySeq = new Map<number, Safety>();
net.on("safety", (m) => {
  if (!viewing) return;
  bySeq.set(m.seq, m);
  bySeq.delete(m.seq - 60);
});
net.onCameraFrame(async (seq, jpeg) => {
  if (chipState.source && !viewing) return;
  const bmp = await createImageBitmap(jpeg);
  if (!viewing) {
    viewing = true;
    started();
  }
  if (disp.width !== bmp.width || disp.height !== bmp.height) {
    disp.width = bmp.width;
    disp.height = bmp.height;
  }
  disp.getContext("2d")!.drawImage(bmp, 0, 0);
  bmp.close();
  viewSeq = seq;
});

function frame(now: number) {
  if (viewing) drawOverlay(overlay, disp.width, disp.height, bySeq.get(viewSeq) ?? latest);
  else drawOverlay(overlay, video.videoWidth, video.videoHeight, syncedFrame(now));
  drawRing(ring, chipState.belt, latest);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function started() {
  $("start").hidden = true;
  $("hud").hidden = false;
  chipState.source = true;
  renderChips();
}
$("btn-camera").addEventListener("click", async () => {
  try {
    await source.startCamera();
    started();
  } catch (e) {
    $("start").querySelector("p")!.textContent = `Camera unavailable: ${(e as Error).message}`;
  }
});
$<HTMLInputElement>("file").addEventListener("change", async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  await source.startFile(f);
  started();
});
$("btn-audio").addEventListener("click", async (ev) => {
  const on = await audio.toggle();
  const b = ev.currentTarget as HTMLButtonElement;
  b.setAttribute("aria-pressed", String(on));
  b.textContent = `Audio cues: ${on ? "on" : "off"}`;
});

// ---------------------------------------------------------------- map
net.onMapChunk((n, p, c) => map.addChunk(n, p, c));
net.onMapReplace((n, p, c) => map.replaceMap(n, p, c));
map.onCount = (n) => ($("map-count").textContent = `${n.toLocaleString()} points`);
net.on("trail", (m) => map.setTrail(m.points));
net.on("pose", (m) => map.setPose(m));
$("btn-fit").addEventListener("click", () => map.fit());

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
}
let hotId: number | null = null;
function renderPins() {
  const list = $("pinlist");
  list.innerHTML = "";
  for (const p of pins.filter((x) => x.kind === "object").sort((a, b) => b.ts - a.ts)) {
    const li = document.createElement("li");
    if (p.id === hotId) li.className = "hot";
    li.innerHTML = `${p.thumb ? `<img alt="" src="data:image/jpeg;base64,${p.thumb}"/>` : ""}<div><b>${p.label}</b><span class="muted small">${ago(p.ts)}${p.near[0] ? " · near " + p.near[0] : ""}</span></div>`;
    li.addEventListener("click", () => select(p));
    list.appendChild(li);
  }
}
function select(p: PinJson) {
  hotId = p.id;
  map.flyTo(p);
  setTab("map");
  renderPins();
}
net.on("pins", (m) => {
  pins = m.pins;
  map.setPins(pins);
  renderPins();
});
setInterval(renderPins, 20_000);

// ---------------------------------------------------------------- asking
function showAnswer(a: Answer) {
  $("answer").hidden = false;
  $("answer-text").textContent = a.text;
  const thumb = $<HTMLImageElement>("answer-thumb");
  thumb.src = a.pin?.thumb ? `data:image/jpeg;base64,${a.pin.thumb}` : "";
  $("answer-meta").textContent = a.pin
    ? `${a.pin.label} · seen ${a.pin.seen}x · ${a.language ?? "template"}`
    : "";
  speak(a.text);
  if (a.pin) select(a.pin);
  else {
    hotId = null;
    map.clearHot();
  }
}
net.on("answer", showAnswer);
$("ask-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = $<HTMLInputElement>("ask-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  net.send({ type: "ask", text });
});
let stopListening: (() => void) | null = null;
$("btn-mic").addEventListener("click", () => {
  const btn = $("btn-mic");
  if (stopListening) {
    stopListening();
    return;
  }
  stopListening = listen(
    (t) => net.send({ type: "ask", text: t }),
    () => {
      stopListening = null;
      btn.classList.remove("on");
    },
  );
  if (stopListening) btn.classList.add("on");
  else $<HTMLInputElement>("ask-input").focus();
});

// ---------------------------------------------------------------- check-ins
let countdown = 0;
let stopAlertListen: (() => void) | null = null;
function closeAlert() {
  $("alert").hidden = true;
  window.clearInterval(countdown);
  stopAlertListen?.();
  stopAlertListen = null;
}
function openAlert(title: string, text: string, ask: boolean) {
  $("alert-title").textContent = title;
  $("alert-text").textContent = text;
  $("alert").hidden = false;
  $("alert-ok").hidden = !ask;
  speak(text);
  if (ask) {
    let left = 30;
    $("alert-count").textContent = `Answering in ${left} s`;
    window.clearInterval(countdown);
    countdown = window.setInterval(() => {
      left -= 1;
      $("alert-count").textContent = left > 0 ? `Answering in ${left} s` : "";
    }, 1000);
    window.setTimeout(() => {
      stopAlertListen = listen(
        (t) => {
          if (/\b(ok|okay|fine|alright|yes|i'm good)\b/i.test(t)) $("alert-ok").click();
        },
        () => {},
      );
    }, 2500);
  } else {
    $("alert-count").textContent = "";
  }
}
net.on("alert", (m) => {
  if (m.kind === "fall_prompt") openAlert("Are you okay?", m.message, true);
  else if (m.kind === "shake_notice") openAlert("Are you okay?", m.message, true);
  else if (m.kind === "fall_escalate") openAlert("No answer", m.message + " Nothing was dialled.", false);
  else if (m.kind === "fall_cancelled") closeAlert();
});
$("alert-ok").addEventListener("click", () => {
  net.send({ type: "fall_ok" });
  closeAlert();
  speak("Okay. Glad you're alright.");
});
$("alert-dismiss").addEventListener("click", closeAlert);
$("btn-test-fall").addEventListener("click", () => void fetch("/api/test_fall", { method: "POST" }));

// ---------------------------------------------------------------- mode
let mode: Mode = "home";
function showMode(m: Mode) {
  mode = m;
  document.documentElement.dataset.mode = m;
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.mode === m));
  });
  map.setFollow(m === "outside");
  $("brand-sub").textContent = m === "home" ? "a memory for your home" : "short-horizon map, safety first";
}
net.on("mode", (m) => showMode(m.mode));
document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.mode !== mode) net.send({ type: "mode", mode: b.dataset.mode });
  });
});

// ---------------------------------------------------------------- tabs
function setTab(tab: string) {
  document.querySelector(".grid")!.setAttribute("data-tab", tab);
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => {
    b.classList.toggle("on", b.dataset.tab === tab);
  });
}
document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => {
  b.addEventListener("click", () => setTab(b.dataset.tab!));
});

setupGlass();
renderChips();
net.connect();
