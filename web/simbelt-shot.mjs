// The sim belt panel as stills: node simbelt-shot.mjs URL T OUT.png [W H] for one moment, or
// node simbelt-shot.mjs URL all OUTDIR [W H] for one PNG per video frame (00001.png ...), the
// cues stepped at 30 fps with the panel easing between them as it does live.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [URL, T, OUT, W = "1200", H = "520"] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--use-gl=angle", "--use-angle=metal"],
});
const page = await browser.newPage({ viewport: { width: +W, height: +H }, deviceScaleFactor: 2 });
page.on("pageerror", (e) => console.log("page error:", e.message));
await page.goto(URL);
await page.waitForFunction(() => typeof window.__show === "function");
await page.waitForTimeout(3500); // GLB parts load
if (T === "all") {
  const { mkdirSync } = await import("node:fs");
  mkdirSync(OUT, { recursive: true });
  const n = await page.evaluate(() => window.__frames());
  for (let k = 0; k < n; k++) {
    await page.evaluate((t) => window.__show(t), k / 30);
    await page.waitForTimeout(33);
    const path = `${OUT}/${String(k + 1).padStart(5, "0")}.png`;
    // a busy GPU (Blender renders beside it) can stall one capture past the default 30 s
    await page
      .screenshot({ path, timeout: 120000 })
      .catch(() => page.screenshot({ path, timeout: 120000 }));
  }
} else {
  await page.evaluate((t) => window.__show(t), +T);
  await page.waitForTimeout(1500); // the cells ease to their levels
  await page.screenshot({ path: OUT });
}
await browser.close();
