import "./styles.css";
import { AudioCues } from "./audio";
import { Source } from "./camera";
import { MapView } from "./map3d";
import { setupGlass } from "./glass";
import { Net } from "./net";
import { Belt3D } from "./belt3d";
import { NaviBot } from "./navibot";
import { close, emoteFor, lineFor, moodOf } from "./navimood";
import { BeltCells, EdgeArcs, type ScreenPin, cellsFromBelt, cellsFromCues, drawOverlay } from "./overlay";
import { streamMic } from "./mic";
import { listen, speak } from "./speech";
import type { Answer, Belt, Device, Mode, PinJson, Safety } from "./types";
import { answerGlow, guideBearing, isPerson, peopleBearings } from "./visual";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const net = new Net();
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
// The belt in 3D: eight motors by default; the demo tour shows the built four and sixteen.
const belt3d = new Belt3D($("belt3d"));
belt3d.onLayout = (n) => ($("btn-motors").textContent = `Belt: ${n} motors`);
$("btn-motors").addEventListener("click", () => belt3d.next());
$("btn-belt-tour").addEventListener("click", () => belt3d.tour());
const belt = new BeltCells($("belt"));
const map = new MapView($<HTMLCanvasElement>("map3d"));
// On wide screens the map is the corner minimap: a floor plan, not an orbit view.
const wide = window.matchMedia("(min-width: 901px)");
map.setTopDown(wide.matches);
wide.addEventListener("change", () => map.setTopDown(wide.matches));
const audio = new AudioCues();
// Navi, the companion on the ask bar: it reacts to everything below through these few fields.
const navi = new NaviBot($("navi-host"));
const naviState = { waitingSince: null as number | null, foundAt: null as number | null, wasFound: false, wasClose: false, lastAct: performance.now(), lastPinCount: -1, lastPinPop: 0, heard: 0 };
const poke = () => (naviState.lastAct = performance.now());
const source = new Source(video, net);
let latest: Safety | null = null;
let pins: PinJson[] = [];
// The pin being guided to (selected or answered), and the edge arcs toward it and toward people.
let target: PinJson | null = null;
const arcs = new EdgeArcs();
/** Guide to a pin, or stop guiding: the ribbon on the map floor and the arc on the screen edge. */
function setGuideTarget(pin: PinJson | null, tell = true) {
  if (tell && (pin?.id ?? null) !== (target?.id ?? null)) net.send({ type: "guide", pin_id: pin?.id ?? null });
  target = pin;
  map.setGuideTarget(pin);
}
/** People or voices around the wearer, as bearings; drawn only when someone reports them. */
function setPeople(list: { bearingDeg: number }[]) {
  arcs.setPeople(list);
}
net.on("people", (m) => setPeople(peopleBearings(m)));

// ---------------------------------------------------------------- status chips
const chipState = {
  server: false,
  memory: "loading",
  finder: "",
  fps: 0,
  source: false,
  belt: null as Belt | null,
  device: null as Device | null,
};
function renderChips() {
  const chip = (cls: string, text: string) => `<span class="chip ${cls}"><i></i>${text}</span>`;
  $("chips").innerHTML = [
    chip(chipState.server ? "ok" : "bad", chipState.server ? "server" : "offline"),
    chip(chipState.memory === "ready" ? "ok" : "", chipState.memory === "ready" ? "memory" : "memory loading"),
    chipState.source ? chip("live", `${chipState.fps.toFixed(0)} fps`) : "",
    chipState.device ? deviceChip(chipState.device) : "",
    chipState.belt
      ? chipState.belt.lost
        ? chip("bad", `${chipState.belt.device}: link lost`)
        : chip("ok", `${chipState.belt.device} · ${chipState.belt.latency_ms?.toFixed(0) ?? "-"} ms`)
      : chip("", "belt offline"),
  ].join("");
}
function deviceChip(d: Device) {
  const on = d.state === "streaming";
  const lat = d.latency_ms === undefined ? "" : ` · ${d.latency_ms} ms`;
  const text = on ? `${d.name} · ${d.res[0]}×${d.res[1]} @ ${d.fps}${lat}` : `${d.name}: ${d.state}`;
  return `<span class="chip ${on ? "ok" : ""}" title="IMU ${d.imu_hz} Hz · link ${d.kbps} of ${d.budget_kbps} kbps"><i></i>${text}</span>`;
}
net.on("device", (m) => {
  chipState.device = m;
  renderChips();
});
// What the glasses hear, typed into the ask box as it is spoken; the final question is sent by
// the glasses themselves.
net.on("transcript", (m) => {
  naviState.heard = m.final ? 0 : performance.now();
  poke();
  if (m.final && m.text.trim()) asked();
  $<HTMLInputElement>("ask-input").value = m.final ? "" : m.text;
});
let everUp = false;
net.onConnection((up) => {
  if (up && !everUp) navi.say(lineFor("hello"), 5200);
  else if (!up && chipState.server) navi.say(lineFor("offline"));
  everUp ||= up;
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
  if (m.found && target) setGuideTarget(null);
  if (m.found && !naviState.wasFound) {
    naviState.foundAt = performance.now();
    navi.emote(emoteFor("found"));
    navi.say(lineFor("found", m.guide?.label ?? undefined));
    poke();
  }
  naviState.wasFound = m.found;
  const near = close(m);
  if (near && !naviState.wasClose) navi.emote(emoteFor("obstacle"));
  naviState.wasClose = near;
  fpv.setLiveYaw(m.yaw ?? null);
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

// Remembered things in view, as halos over the camera, from the live first-person pose.
function screenPins(now: number): ScreenPin[] {
  const glow = hotAt === null ? null : answerGlow(now - hotAt);
  return fpv.project(pins.filter((p) => p.kind === "object")).map(({ pin, x, y, dist }) => ({
    label: pin.label,
    x,
    y,
    dist,
    selected: pin.id === hotId,
    bright: pin.id === hotId ? (glow?.bright ?? 0) : 0,
    ring: pin.id === hotId ? (glow?.ring ?? null) : null,
  }));
}

function frame(now: number) {
  const mine = target ? (fpv.bearingTo(target.pos) ?? map.bearingTo(target.pos)) : null;
  const bearing = latest?.guide ? guideBearing(latest.guide, mine) : mine;
  arcs.setGuide(bearing);
  const extra = { pins: screenPins(now), arcs };
  if (viewing) drawOverlay(overlay, disp.width, disp.height, bySeq.get(viewSeq) ?? latest, false, extra);
  else drawOverlay(overlay, video.videoWidth, video.videoHeight, syncedFrame(now), false, extra);
  // The belt's own telemetry when one is connected, otherwise what the server would send it.
  const b = chipState.belt;
  const cells = b ? cellsFromBelt(b) : cellsFromCues(latest, now);
  belt3d.set(cells, bearing ?? null);
  if (naviState.waitingSince !== null && now - naviState.waitingSince > 15_000) naviState.waitingSince = null;
  navi.set(
    moodOf({
      online: chipState.server,
      listening: stopListening !== null || now - naviState.heard < 2000,
      waiting: naviState.waitingSince !== null,
      alert: !$("alert").hidden,
      ttc: latest?.ttc ?? null,
      path: latest?.path ?? null,
      bearing: bearing ?? null,
      sinceFoundMs: naviState.foundAt === null ? null : now - naviState.foundAt,
      quietMs: now - naviState.lastAct,
    }),
    bearing ?? null,
    Math.max(0, ...cells.map((c) => c.level)),
  );
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
// The other main view: the same map seen through the camera, from the live pose. The camera
// itself is the default.
const fpvCanvas = document.createElement("canvas");
fpvCanvas.id = "fpv";
stage.insertBefore(fpvCanvas, overlay);
const fpv = new MapView(fpvCanvas);
fpv.setFirstPerson(true);
// The camera is the canvas; everything floats over it. The map view is one tap away.
let fpvOn = false;
let frameAspect = 9 / 16;
function showView() {
  const on = fpvOn && fpv.hasPose;
  stage.classList.toggle("fpv", on);
  $("btn-fpv").textContent = fpvOn ? "View: map" : "View: camera";
  if (!on) return;
  // The map is drawn in the shape of the frame and fills the screen, cropped like the video.
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const k = Math.max(w / frameAspect, h);
  fpvCanvas.style.width = `${Math.round(k * frameAspect)}px`;
  fpvCanvas.style.height = `${Math.round(k)}px`;
}
$("btn-fpv").addEventListener("click", () => {
  fpvOn = !fpvOn;
  showView();
});
window.addEventListener("resize", showView);
net.onMapChunk((n, p, c, z) => {
  map.addChunk(n, p, c, z);
  fpv.addChunk(n, p, c, z);
  showView();
});
net.onMapReplace((n, p, c, z) => {
  map.replaceMap(n, p, c, z);
  fpv.replaceMap(n, p, c, z);
  showView();
});
map.onCount = (n) => ($("map-count").textContent = `${n.toLocaleString()} points`);
net.on("trail", (m) => {
  map.setTrail(m.points);
  fpv.setTrail(m.points);
});
net.on("pose", (m) => {
  map.setPose(m);
  fpv.setPose(m);
  if (m.aspect) frameAspect = m.aspect;
  showView();
});
$("btn-fit").addEventListener("click", () => map.fit());

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
}
let hotId: number | null = null;
let hotAt: number | null = null; // when the selected pin answered a question
function renderPins() {
  const list = $("pinlist");
  list.innerHTML = "";
  for (const p of pins.filter((x) => x.kind === "object" && !isPerson(x.label)).sort((a, b) => b.ts - a.ts)) {
    const li = document.createElement("li");
    if (p.id === hotId) li.className = "hot";
    li.innerHTML = `${p.thumb ? `<img alt="" src="data:image/jpeg;base64,${p.thumb}"/>` : ""}<div><b>${p.label}</b><span class="muted small">${ago(p.ts)}${p.near[0] ? " · near " + p.near[0] : ""}</span></div>`;
    li.addEventListener("click", () => select(p));
    list.appendChild(li);
  }
}
function select(p: PinJson, answer = false) {
  hotId = p.id;
  hotAt = answer ? performance.now() : null;
  map.flyTo(p, answer);
  setGuideTarget(p);
  setTab("map");
  renderPins();
}
map.onPick = (p) => select(p);
net.on("pins", (m) => {
  // A new thing remembered: a little surprise, at most every eight seconds.
  const n = m.pins.filter((p) => p.kind === "object").length;
  const t = performance.now();
  if (naviState.lastPinCount >= 0 && n > naviState.lastPinCount && t - naviState.lastPinPop > 8000) {
    navi.emote(emoteFor("pin"));
    naviState.lastPinPop = t;
  }
  naviState.lastPinCount = n;
  pins = m.pins;
  map.setPins(pins);
  renderPins();
});
setInterval(renderPins, 20_000);

// ---------------------------------------------------------------- asking
function showAnswer(a: Answer) {
  if (a.cue) {
    // A route cue on the way ("Door ahead."): spoken, and the guide target stays.
    speak(a.text);
    return;
  }
  $("answer").hidden = false;
  $("answer-text").textContent = a.text;
  const thumb = $<HTMLImageElement>("answer-thumb");
  thumb.src = a.pin?.thumb ? `data:image/jpeg;base64,${a.pin.thumb}` : "";
  thumb.hidden = !a.pin?.thumb;
  // How long ago, always, from the newest sighting.
  $("answer-meta").textContent = a.pin
    ? `${a.pin.label} · ${a.live ? "in view now" : `seen ${ago(a.pin.ts)}`} · ${a.pin.seen}x`
    : a.live && a.label
      ? `${a.label} · in view now`
      : "";
  $("answer-meta").title = a.language ?? "template";
  speak(a.text);
  naviState.waitingSince = null;
  poke();
  const label = a.pin?.label ?? (a.live ? a.label : null);
  navi.emote(emoteFor(label ? "answer" : "answer-none"));
  navi.say(lineFor(label ? "answer" : "answer-none", label ?? undefined));
  if (a.pin) select(a.pin, true);
  else {
    hotId = null;
    hotAt = null;
    map.clearHot();
    setGuideTarget(null, false); // a live sighting has no pin; the server guides to it itself
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
  asked();
});
let stopListening: (() => void) | null = null;
// Muse Voice Transcribe on the server: the mic stays open and the model finds each question's end.
let serverAsr: Promise<boolean> = fetch("/api/state")
  .then((r) => r.json())
  .then((s) => Boolean(s.asr))
  .catch(() => false);
/** A question went out: Navi thinks until the answer comes back. */
function asked() {
  naviState.waitingSince = performance.now();
  poke();
  navi.say(lineFor("thinking"), 2500);
}
$("btn-mic").addEventListener("click", async () => {
  poke();
  const btn = $("btn-mic");
  if (stopListening) {
    stopListening();
    stopListening = null;
    btn.classList.remove("on");
    return;
  }
  if (await serverAsr) {
    stopListening = await streamMic(net);
    if (stopListening) {
      btn.classList.add("on");
      navi.say(lineFor("listening"), 2500);
      return;
    }
    serverAsr = Promise.resolve(false);
  }
  stopListening = listen(
    (t) => {
      net.send({ type: "ask", text: t });
      asked();
    },
    (failed) => {
      stopListening = null;
      btn.classList.remove("on");
      if (failed) $<HTMLInputElement>("ask-input").focus(); // nothing heard: type it instead
    },
  );
  if (stopListening) {
    btn.classList.add("on");
    navi.say(lineFor("listening"), 2500);
  }
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
  if (m.kind === "fall_prompt" || m.kind === "shake_notice") navi.emote(emoteFor("alert"));
  if (m.kind === "fall_prompt") openAlert("Are you okay?", m.message, true);
  else if (m.kind === "shake_notice") openAlert("Are you okay?", m.message, true);
  else if (m.kind === "fall_escalate") openAlert("No answer", m.message + " Nothing was dialled.", false);
  else if (m.kind === "fall_cancelled") closeAlert();
  else if ((m.kind === "moved" || m.kind === "maybe_moved") && m.message) speak(m.message);
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

// ---------------------------------------------------------------- details
// Debug readouts (status chips, the safety readout, the point count) live behind this toggle, so
// the default screen is the camera, the blob and the ring.
$("btn-details").addEventListener("click", (ev) => {
  const on = document.documentElement.dataset.details !== "on";
  document.documentElement.dataset.details = on ? "on" : "";
  (ev.currentTarget as HTMLElement).setAttribute("aria-pressed", String(on));
});

setupGlass();
renderChips();
net.connect();
