// Record one demo-capture scene in real time: real Chrome, live tab capture + MediaRecorder,
// never frame-stepped. Writes showcase/capture/<scene>-<timestamp>.mp4 (h264 high, yuv420p,
// faststart).
//
// Usage: node scripts/capture.mjs <obstacle|recall|navigate|map|belt|split>
//          [--server http://127.0.0.1:8445]   live feed (default): the glasses through the service
//          [--walk data/walks/<id>]           or a saved walk, replayed at its own timing
//          [--video path.mp4]                 or a plain video, cut into a walk first
//          [--from 0 (video only)] [--secs 15] [--fps 30|60] [--vertical] [--ask keys] [--ask-at 6]
//          [--port 8451] [--hfov 70 (video only)] [--warm 0] [--label-radius 2] [--hold-answer 0] [--native] [--headed] [--label name]
// Two passes: the server's stream to the page is logged first (its GPU models stall Chrome's
// paints), then the server stops and the page is recorded while the log plays back at its own
// timing. --secs 0 logs until Ctrl-C, then records the whole log. --warm waits that many seconds after the feed starts (a
// replayed map grows as the walk plays, so give the map scene 30 to 60). A replay runs scripts/judge_replay.py on --port with a copy of
// the house, so the live service is never written. Clips from data/pov (third-party footage)
// get TEST in the file name and must not be published. --native keeps the 2x capture size.
// Chrome runs headless by default (the new headless: the same browser, compositor and real-time
// clock): a headed window bigger than the screen gets clipped, and the mini's screen is 1080p.
// --headed shows the window, for a screen larger than the stage.
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { chromium } = createRequire(join(ROOT, "web/package.json"))("playwright");
const SCENES = ["obstacle", "recall", "navigate", "map", "belt", "split"];

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const scene = argv[0];
if (!SCENES.includes(scene)) {
  console.error(`scene must be one of ${SCENES.join(", ")}`);
  process.exit(2);
}
const walkArg = opt("walk");
const videoArg = opt("video");
const port = Number(opt("port", 8451));
const secs = Number(opt("secs", 15));
const fps = Number(opt("fps", 30));
const vertical = flag("vertical");
const [W, H] = vertical ? [1080, 1920] : [1920, 1080];
const from = Number(opt("from", 0));
const dpr = Number(opt("dpr", 2));
const replay = Boolean(walkArg || videoArg);
const thirdParty = [walkArg, videoArg].some((p) => p && resolve(p).includes("/data/pov/"));

// ---------------------------------------------------------------- gates and the heavy-job slot
function gates() {
  const mp = spawnSync("memory_pressure", { encoding: "utf8" }).stdout ?? "";
  const free = Number(/free percentage: (\d+)%/.exec(mp)?.[1] ?? 100);
  const df = spawnSync("df", ["-k", ROOT], { encoding: "utf8" }).stdout.trim().split("\n").pop().split(/\s+/);
  const gib = Number(df[3]) / 1048576;
  if (free < 10) throw new Error(`memory free ${free}% is under 10%; not recording`);
  if (gib < 15) throw new Error(`disk free ${gib.toFixed(1)} GiB is under 15; not recording`);
}
function takeSlot(what) {
  for (let n = 1; ; n++) {
    const slot = join(homedir(), "helloworld", n === 1 ? ".heavy-lock" : `.heavy-lock-${n}`);
    try {
      mkdirSync(slot);
    } catch {
      let pid = 0;
      try {
        pid = Number(readFileSync(join(slot, "owner"), "utf8").split(/\s/)[0]);
        process.kill(pid, 0);
        continue;
      } catch {
        if (!pid) continue; // owner not written yet, or no owner file: leave it to its writer
        rmSync(slot, { recursive: true, force: true });
        n--;
        continue;
      }
    }
    writeFileSync(join(slot, "owner"), `${process.pid} ${what}\n`);
    return slot;
  }
}

// ---------------------------------------------------------------- a walk the replay can play
/** A saved walk replays as it is (judge_replay.py fills hfov itself). A plain video has no frames
 *  yet: cut it into a temp walk from --from on. */
function videoWalk(tmp) {
  const out = join(tmp, "walk");
  mkdirSync(out);
  const src = resolve(videoArg);
  mkdirSync(join(out, "frames"));
  const r = spawnSync("nice", ["-n", "19", "taskpolicy", "-b", "ffmpeg", "-loglevel", "error", "-ss", String(from), "-i", src,
    "-t", String((secs || 120) + 40), "-vf", "fps=15", "-q:v", "3", join(out, "frames", "%05d.jpg")], { stdio: "inherit" });
  if (r.status !== 0) throw new Error("ffmpeg could not cut the video into frames");
  const [w, h] = spawnSync("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", src],
    { encoding: "utf8" }).stdout.trim().split(",").map(Number);
  const n = spawnSync("ls", [join(out, "frames")], { encoding: "utf8" }).stdout.trim().split("\n").length;
  const t0 = Date.now();
  const lines = Array.from({ length: n }, (_, i) =>
    JSON.stringify({ file: `${String(i + 1).padStart(5, "0")}.jpg`, ts: t0 + (i * 1000) / 15, seq: i }));
  writeFileSync(join(out, "frames.jsonl"), lines.join("\n") + "\n");
  writeFileSync(join(out, "camera.json"), JSON.stringify({ source: "video", width: w, height: h, hfov: Number(opt("hfov", 70)) }));
  return out;
}

// ---------------------------------------------------------------- run
const tmp = mkdtempSync(join(tmpdir(), "ms-capture-"));
const d = new Date();
const two = (n) => String(n).padStart(2, "0");
const stamp = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
const label = opt("label", "") ? `-${opt("label")}` : "";
const name = `${scene}${vertical ? "-vertical" : ""}${label}${thirdParty ? "-TEST" : ""}-${stamp}`;
const outDir = join(ROOT, "showcase", "capture");
mkdirSync(outDir, { recursive: true });
const raw = join(tmp, `${name}.webm`);
const mp4 = join(outDir, `${name}.mp4`);

let slot = null;
let feeder = null;
let browser = null;
let stopping = false;
process.on("SIGINT", () => (stopping = true));
process.on("SIGTERM", () => (stopping = true));

function cleanup() {
  if (feeder && feeder.exitCode === null) feeder.kill("SIGTERM"); // the replay frees its own server and slot
  if (slot) rmSync(slot, { recursive: true, force: true });
  rmSync(tmp, { recursive: true, force: true });
}

try {
  gates();
  slot = takeSlot(`capture ${scene}`);
  let base = opt("server", "http://127.0.0.1:8445");
  if (replay) {
    const walk = walkArg ? resolve(walkArg) : videoWalk(tmp);
    base = `http://127.0.0.1:${port}`;
    // Enough laps that the walk outlasts the take.
    feeder = spawn(join(ROOT, ".venv/bin/python"), ["-u", "scripts/judge_replay.py", walk, "--port", String(port),
      "--ask", "", "--settle", "4", "--laps", "4"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
    const log = (b) => process.stdout.write(String(b).replace(/^/gm, "  replay | "));
    feeder.stdout.on("data", log);
    feeder.stderr.on("data", log);
    console.log(`replay pid ${feeder.pid} on :${port}`);
  }
  const params = new URLSearchParams({ capture: scene });
  if (vertical) params.set("vertical", "1");
  if (opt("ask")) params.set("ask", opt("ask"));
  if (opt("ask-at")) params.set("ask-at", opt("ask-at"));
  if (opt("period")) params.set("period", opt("period"));
  if (opt("hold-answer")) params.set("hold-answer", opt("hold-answer"));
  if (opt("label-radius")) params.set("label-radius", opt("label-radius"));
  const url = `${base}/capture.html?${params}`;

  const launch = { headless: !flag("headed"), args: ["--auto-accept-this-tab-capture", "--autoplay-policy=no-user-gesture-required",
    "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
    // Software compositing: the GPU compositor in headless Chrome missed 3 to 5 vsyncs a few
    // times a second whenever a canvas animated; WebGL still draws on the GPU.
    "--disable-gpu-compositing", "--use-gl=angle", "--use-angle=metal", `--window-size=${W},${H + 90}`] };
  try {
    browser = await chromium.launch({ ...launch, channel: "chrome" });
  } catch {
    browser = await chromium.launch({ ...launch, executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" });
  }
  const ready = scene === "map" ? "s.points > 0" : ["recall", "navigate", "split"].includes(scene) ? "s.frames > 0 && s.points > 0" : "s.frames > 0";
  const waitReady = (pg) => pg.waitForFunction(`(() => { const s = window.__capture; return s && ${ready}; })()`, null, { timeout: 300_000, polling: 250 });
  const settle = (scene === "obstacle" || scene === "belt" ? 1.2 : 3) + Number(opt("warm", 0)); // the map fit, the first overlay
  const sleepUntil = (t) => new Promise((r) => {
    const tick = () => (stopping || Date.now() >= t ? r() : setTimeout(tick, 100));
    tick();
  });

  // Pass 1, the feed: the server's stream to the page, logged as it arrives, nothing recorded. The
  // server's models share the GPU with Chrome's compositor, and paints stalled 50 to 100 ms while
  // they ran, so the take itself is made in pass 2 with the server already stopped.
  const feed = [];
  {
    const ctx = await browser.newContext({ viewport: { width: W, height: H } });
    let logging = true;
    await ctx.routeWebSocket(/\/ws$/, (ws) => {
      const server = ws.connectToServer();
      const t0 = performance.now();
      const log = logging; // only the first connection; a reconnect just passes through
      logging = false;
      server.onMessage((m) => {
        if (log) feed.push([performance.now() - t0, m]);
        ws.send(m);
      });
    });
    const pg = await ctx.newPage();
    pg.on("pageerror", (e) => console.log("page error:", e.message));
    // The server can take minutes to load its models; keep trying until the page is served.
    for (let t = Date.now(); ; ) {
      try {
        await pg.goto(url, { timeout: 10_000 });
        break;
      } catch (e) {
        if (stopping || Date.now() - t > 300_000) throw e;
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    console.log(`logging the feed from ${url}`);
    await waitReady(pg);
    await sleepUntil(secs ? Date.now() + (settle + secs + 1.5) * 1000 : Infinity);
    await ctx.close();
  }
  if (feeder && feeder.exitCode === null) {
    feeder.kill("SIGTERM"); // the replay frees its own server and slot
    await new Promise((r) => feeder.once("exit", r));
  }
  // Skip the wait while the server loaded its models: what came before the first camera frame
  // (the saved map, the pins) goes out at once.
  const first = feed.findIndex(([, m]) => typeof m !== "string" && m[0] === 70);
  if (first < 0) throw new Error("the feed sent no camera frames");
  const lead = Math.max(0, feed[first][0] - 500);
  for (const f of feed) f[0] = Math.max(0, f[0] - lead);
  // The server hands out a frame whenever it finishes one, in bursts and stalls. Space the camera
  // frames evenly at their average rate, each frame's safety message moved with it, so the take
  // never holds one picture longer than the feed's own pace.
  const isFrame = (m) => typeof m !== "string" && m[0] === 70;
  const frames = feed.filter(([, m]) => isFrame(m));
  const span = frames.at(-1)[0] - frames[0][0];
  const at = new Map(frames.map((f, i) => [f[1].readUInt32LE(1), frames[0][0] + (i * span) / Math.max(1, frames.length - 1)]));
  let worst = 0;
  frames.forEach((f, i) => i && (worst = Math.max(worst, f[0] - frames[i - 1][0])));
  for (const f of feed) {
    const m = typeof f[1] === "string" ? JSON.parse(f[1]) : null;
    const seq = isFrame(f[1]) ? f[1].readUInt32LE(1) : m?.type === "safety" ? m.seq : undefined;
    if (at.has(seq)) f[0] = at.get(seq);
  }
  feed.sort((a, b) => a[0] - b[0]);
  console.log(`feed: ${frames.length} camera frames, ${(1000 / (span / (frames.length - 1))).toFixed(1)} fps, longest server gap ${worst.toFixed(0)} ms, now evened out`);
  stopping = false;

  // Pass 2, the take: the same page, served from web/dist, gets the logged stream at its logged
  // times, in real time, and the tab is recorded.
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: dpr });
  await ctx.route(`${base}/**`, (route) => {
    const path = new URL(route.request().url()).pathname;
    const f = join(ROOT, "web/dist", path === "/" ? "index.html" : path);
    return existsSync(f) ? route.fulfill({ path: f }) : route.fulfill({ status: 404 });
  });
  let feedStart = 0;
  await ctx.routeWebSocket(/\/ws$/, (ws) => {
    ws.onMessage(() => {}); // the page's own ask was answered in the feed
    if (feedStart) return;
    feedStart = Date.now();
    const t0 = performance.now();
    let i = 0;
    const pump = () => {
      while (i < feed.length && feed[i][0] <= performance.now() - t0) ws.send(feed[i++][1]);
      if (i < feed.length) setTimeout(pump, Math.max(0, feed[i][0] - (performance.now() - t0)));
    };
    pump();
  });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.log("page error:", e.message));
  const file = createWriteStream(raw);
  await page.goto(url);
  await waitReady(page);
  await page.waitForTimeout(settle * 1000);

  await page.evaluate(async ([w, h, rate, k]) => {
    const s = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: rate, width: { ideal: w * k }, height: { ideal: h * k } },
      audio: false,
      preferCurrentTab: true,
    });
    // Chrome sometimes opens the stream at 1x; ask again for the 2x surface.
    const track = s.getVideoTracks()[0];
    if ((track.getSettings().width ?? 0) < w * k) await track.applyConstraints({ frameRate: rate, width: w * k, height: h * k }).catch(() => {});
    // H.264 goes to the Mac's hardware encoder; a software VP9 encode of a 4K tab fell behind.
    const mimeType = ["video/webm;codecs=h264", "video/webm;codecs=vp9"].find((t) => MediaRecorder.isTypeSupported(t));
    const rec = new MediaRecorder(s, { mimeType, videoBitsPerSecond: 40_000_000 });
    // Chunks stay in the page until the take ends: encoding them for the trip out on the main
    // thread mid-take stalled a paint every second.
    const chunks = (window.__chunks = []);
    rec.ondataavailable = (e) => chunks.push(e.data);
    window.__rec = rec;
    // Page-side frame pacing over the take: a throttled renderer shows up here, not in the file.
    const raf = (window.__raf = { n: 0, max: 0, last: 0, slow: 0 });
    const beat = (t) => {
      if (raf.last) raf.max = Math.max(raf.max, t - raf.last);
      if (raf.last && t - raf.last > 50) raf.slow++;
      raf.last = t;
      raf.n++;
      if (rec.state !== "inactive") requestAnimationFrame(beat);
    };
    requestAnimationFrame(beat);
    window.__capture.frameGap = 0;
    rec.start(1000);
    const st = track.getSettings();
    return `${st.width}x${st.height} @ ${st.frameRate}, ${mimeType}`;
  }, [W, H, fps, dpr]).then((size) => console.log(`tab capture ${size}`));
  console.log(secs ? `recording ${secs} s` : "recording to the end of the feed; Ctrl-C to stop");
  // Half a second more than asked: the encode drops the first 0.5 s, where the capture can hitch.
  await sleepUntil(secs ? Date.now() + (secs + 0.5) * 1000 : feedStart + feed.at(-1)[0]);
  const state = await page.evaluate(() => new Promise((r) => {
    window.__rec.onstop = () => r({ ...window.__capture, raf: window.__raf });
    window.__rec.stop();
  }));
  const parts = await page.evaluate(() => window.__chunks.length);
  for (let i = 0; i < parts; i++) {
    const b64 = await page.evaluate(async (k) => {
      const buf = new Uint8Array(await window.__chunks[k].arrayBuffer());
      window.__chunks[k] = null;
      let bin = "";
      for (let j = 0; j < buf.length; j += 0x8000) bin += String.fromCharCode(...buf.subarray(j, j + 0x8000));
      return btoa(bin);
    }, i);
    file.write(Buffer.from(b64, "base64"));
  }
  await new Promise((r) => file.end(r));
  await browser.close();
  browser = null;
  console.log(`page saw ${state.frames} camera frames, ${state.points} map points, ${state.answers} answers; ` +
    `${state.raf.n} paints, longest ${state.raf.max.toFixed(0)} ms, ${state.raf.slow} over 50 ms; longest camera wait ${state.frameGap.toFixed(0)} ms`);
  const pts = spawnSync("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0", raw],
    { encoding: "utf8" }).stdout.trim().split("\n").map(Number).filter((x) => !Number.isNaN(x));
  const gap = pts.slice(1).reduce((m, t, i) => Math.max(m, t - pts[i]), 0);
  console.log(`recorder wrote ${pts.length} frames over ${(pts.at(-1) - pts[0]).toFixed(1)} s, longest gap ${(gap * 1000).toFixed(0)} ms`);

  const scale = flag("native") ? [] : ["-vf", `scale=${W}:${H}:flags=lanczos`];
  const enc = spawnSync("nice", ["-n", "19", "taskpolicy", "-b", "ffmpeg", "-loglevel", "error", "-y", "-ss", "0.5", "-i", raw, ...scale,
    "-fps_mode", "cfr", "-r", String(fps), "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "16",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4], { stdio: "inherit" });
  if (enc.status !== 0) throw new Error("ffmpeg encode failed");
  console.log(`wrote ${mp4}`);
} catch (e) {
  console.error(String(e?.message ?? e));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  cleanup();
}
