// The HUD's spoken ask, end to end against a running server: a stand-in for the browser's
// speech recognition hears one sentence, the mic button sends it, and the answer and the guide
// come back from the real server.
//   node voice-e2e.mjs http://127.0.0.1:8530 "take me to my bed"
import { chromium } from "playwright";
const url = process.argv[2] ?? "http://127.0.0.1:8445";
const said = process.argv[3] ?? "take me to my bed";
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("page error:", e.message));
await page.addInitScript((text) => {
  class FakeRecognition {
    start() {
      setTimeout(() => {
        this.onresult?.({ results: [[{ transcript: text }]] });
        this.onend?.();
      }, 300);
    }
    stop() {}
  }
  window.webkitSpeechRecognition = FakeRecognition;
  // Every message the page sends, to check what the mic button put on the wire.
  const send = WebSocket.prototype.send;
  window.__sent = [];
  WebSocket.prototype.send = function (d) {
    if (typeof d === "string") window.__sent.push(d);
    return send.call(this, d);
  };
}, said);
await page.goto(url);
await page.waitForTimeout(3000);
await page.click("#btn-mic");
await page.waitForFunction(() => document.querySelector("#answer-text")?.textContent, null, { timeout: 20000 });
await page.waitForTimeout(2500);
const asks = (await page.evaluate(() => window.__sent)).filter((s) => s.includes('"ask"'));
console.log("sent:", asks.join(" "));
console.log("answer:", await page.textContent("#answer-text"));
console.log("readout:", await page.textContent("#readout"));
await browser.close();
