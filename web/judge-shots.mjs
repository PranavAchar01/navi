// Stills of the judge view while a walk plays: node judge-shots.mjs <url> <out-prefix> <secs,...>
// Seconds count from page load; each still is 1920x1080 unless W and H are set.
import { chromium } from "playwright";
const [url, prefix, at = "10,30,50,60,75"] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=metal", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: Number(process.env.W ?? 1920), height: Number(process.env.H ?? 1080) }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.log("page error:", e.message));
const t0 = Date.now();
await page.goto(url);
for (const s of at.split(",").map(Number)) {
  await page.waitForTimeout(Math.max(0, t0 + s * 1000 - Date.now()));
  const out = `${prefix}-${String(s).padStart(3, "0")}s.png`;
  await page.screenshot({ path: out });
  const steps = await page.$$eval("#steps li.done", (l) => l.map((x) => x.textContent));
  console.log(out, "|", steps.join(" "), "|", await page.textContent("#source"));
}
await browser.close();
