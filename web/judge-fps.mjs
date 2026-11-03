// Frame rate of the judge view during a replay: every requestAnimationFrame delta for `secs`
// seconds after a `wait`, then FPS, frame-time percentiles and long tasks. Prints the WebGL
// renderer so a software-GPU number is never mistaken for the real one.
//
//   node judge-fps.mjs http://127.0.0.1:5190/judge.html [wait=10] [secs=30]

import { chromium } from "playwright";

const [url, wait = "10", secs = "30"] = process.argv.slice(2);
const browser = await chromium.launch({ channel: "chromium", args: ["--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(url);
await page.waitForTimeout(Number(wait) * 1000);
const r = await page.evaluate(async (ms) => {
  const gl = document.createElement("canvas").getContext("webgl");
  const dbg = gl?.getExtension("WEBGL_debug_renderer_info");
  const gpu = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "unknown";
  const long = [];
  new PerformanceObserver((l) => l.getEntries().forEach((e) => long.push(e.duration))).observe({ type: "longtask", buffered: false });
  const d = [];
  await new Promise((done) => {
    const t0 = performance.now();
    let last = t0;
    const step = (t) => {
      d.push(t - last);
      last = t;
      if (t - t0 < ms) requestAnimationFrame(step);
      else done();
    };
    requestAnimationFrame(step);
  });
  d.shift();
  const s = [...d].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const total = d.reduce((a, b) => a + b, 0);
  return {
    gpu,
    frames: d.length,
    fps: +((1000 * d.length) / total).toFixed(1),
    p50: +q(0.5).toFixed(1),
    p95: +q(0.95).toFixed(1),
    p99: +q(0.99).toFixed(1),
    max: +s[s.length - 1].toFixed(1),
    over33: d.filter((x) => x > 33.4).length,
    longTasks: long.length,
    longMs: Math.round(long.reduce((a, b) => a + b, 0)),
  };
}, Number(secs) * 1000);
console.log(JSON.stringify(r));
await browser.close();
