import "./styles.css";
import { AudioCues } from "./audio";
import { Source } from "./camera";
import { MapView } from "./map3d";
import { Net } from "./net";
import { BeltCells, drawOverlay, drawRing } from "./overlay";
import { listen, speak } from "./speech";
import type { Answer, PinJson, Safety } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const net = new Net();
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const ring = $<HTMLCanvasElement>("ring");
const belt = new BeltCells($("belt"));
const map = new MapView($<HTMLCanvasElement>("map3d"));
const audio = new AudioCues();
const source = new Source(video, net);
let latest: Safety | null = null;
let pins: PinJson[] = [];

// ---------------------------------------------------------------- status chips
const chipState = { server: false, memory: "loading", finder: "", fps: 0, source: false };
function renderChips() {
  const chip = (cls: string, text: string) => `<span class="chip ${cls}"><i></i>${text}</span>`;
  $("chips").innerHTML = [
    chip(chipState.server ? "ok" : "bad", chipState.server ? "server" : "offline"),
    chip(chipState.memory === "ready" ? "ok" : "", chipState.memory === "ready" ? "memory" : "memory loading"),
    chipState.source ? chip("live", `${chipState.fps.toFixed(0)} fps`) : "",
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
net.on("safety", (m) => {
  latest = m;
  chipState.fps = m.fps;
  belt.update(m);
  audio.update(m);
  $("readout").textContent =
    `depth ${m.ms.depth?.toFixed(0)} ms · detect ${m.ms.detect?.toFixed(0)} ms · scale ${m.scale}` +
    (m.map_age !== null ? ` · map ${m.map_age.toFixed(0)} s old` : "");
  renderChips();
});
function frame() {
  drawOverlay(overlay, video, latest);
  drawRing(ring, latest);
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

renderChips();
net.connect();
