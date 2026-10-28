import { SECTOR_NAMES, type Safety } from "./types";

const COLORS = { ok: "#5ee6a0", warn: "#ffd166", near: "#ff6b6b" };

function toneFor(d: number | null): string {
  if (d === null) return COLORS.ok;
  return d < 1 ? COLORS.near : d < 2 ? COLORS.warn : COLORS.ok;
}

// Proximity glow: every pixel the depth model puts within 1.6 m lights up, yellow from 1.6 m in,
// red inside 0.9 m, the floor left out. Values ease between results so the glow breathes rather
// than flickers, and it is drawn blurred so it reads as light on the scene, not a mask.
const YELLOW = [255, 196, 64];
const RED = [255, 64, 72];
const glow = document.createElement("canvas");
const rim = document.createElement("canvas");
let rimEdge: Float32Array | null = null;
let glowAlpha: Float32Array | null = null;
let glowRed: Float32Array | null = null;
let lastProx = "";

function updateGlow(s: Safety) {
  const [gw, gh] = s.prox_size!;
  if (glow.width !== gw) {
    glow.width = gw;
    glow.height = gh;
    glowAlpha = new Float32Array(gw * gh);
    glowRed = new Float32Array(gw * gh);
    rim.width = gw;
    rim.height = gh;
    rimEdge = new Float32Array(gw * gh);
  }
  const fresh = s.prox! !== lastProx;
  lastProx = s.prox!;
  const raw = Uint8Array.from(atob(s.prox!), (c) => c.charCodeAt(0));
  const img = glow.getContext("2d")!.createImageData(gw, gh);
  const pulse = 0.82 + 0.18 * Math.sin(performance.now() / 160);
  for (let i = 0; i < raw.length; i++) {
    const d = raw[i] === 255 ? 99 : raw[i] / 50;
    if (fresh) {
      // Yellow fades in from 1.6 m, red takes over around 0.9 m (blended over 0.8 to 1.0 m).
      const y = Math.min(1, Math.max(0, (1.6 - d) / 0.6));
      const ta = d >= 1.6 ? 0 : d >= 1.0 ? 0.16 + 0.3 * y * y * (3 - 2 * y) : 0.55 + 0.15 * Math.min(1, 1 - d);
      const tr = d < 0.8 ? 1 : d < 1.0 ? (1.0 - d) / 0.2 : 0;
      glowAlpha![i] += (ta - glowAlpha![i]) * 0.45;
      glowRed![i] += (tr - glowRed![i]) * 0.45;
    }
    const r = glowRed![i];
    const al = glowAlpha![i] * (r > 0.5 ? pulse : 1);
    img.data[i * 4] = YELLOW[0] + (RED[0] - YELLOW[0]) * r;
    img.data[i * 4 + 1] = YELLOW[1] + (RED[1] - YELLOW[1]) * r;
    img.data[i * 4 + 2] = YELLOW[2] + (RED[2] - YELLOW[2]) * r;
    img.data[i * 4 + 3] = Math.min(255, al * 255);
  }
  glow.getContext("2d")!.putImageData(img, 0, 0);
  // Rim: the near side of a depth step (an object's silhouette against what is behind it).
  const rimImg = rim.getContext("2d")!.createImageData(gw, gh);
  for (let yy = 1; yy < gh - 1; yy++) {
    for (let xx = 1; xx < gw - 1; xx++) {
      const i = yy * gw + xx;
      const d = raw[i];
      let e = 0;
      if (d < 80) {
        const m = Math.max(raw[i - 1], raw[i + 1], raw[i - gw], raw[i + gw]);
        e = Math.min(1, Math.max(0, (m - d - 3) / 6)); // a step of 6 cm starts it, 18 cm is full
      }
      if (fresh) rimEdge![i] += (e - rimEdge![i]) * 0.5;
      const a = rimEdge![i] * Math.max(0.6, glowAlpha![i] * 1.8);
      const r = glowRed![i];
      rimImg.data[i * 4] = 255;
      rimImg.data[i * 4 + 1] = 230 + (90 - 230) * r;
      rimImg.data[i * 4 + 2] = 150 + (90 - 150) * r;
      rimImg.data[i * 4 + 3] = Math.min(255, a * 255);
    }
  }
  rim.getContext("2d")!.putImageData(rimImg, 0, 0);
}

/** The proximity glow over the live video. */
export function drawOverlay(canvas: HTMLCanvasElement, video: HTMLVideoElement, s: Safety | null) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!s || !video.videoWidth || !s.prox) return;
  // The video is object-fit: contain, so find where it actually sits in the canvas.
  const scale = Math.min(cw / video.videoWidth, ch / video.videoHeight);
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  const ox = (cw - w) / 2;
  const oy = (ch - h) / 2;
  updateGlow(s);
  ctx.imageSmoothingQuality = "high";
  ctx.filter = `blur(${Math.round(w / 140)}px)`;
  ctx.drawImage(glow, ox, oy, w, h);
  // Rim light: a wide soft halo and a thin bright line on top, both from the same edge map.
  ctx.filter = `blur(${Math.round(w / 220)}px)`;
  ctx.drawImage(rim, ox, oy, w, h);
  ctx.filter = `blur(${Math.max(1, Math.round(w / 900))}px)`;
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(rim, ox, oy, w, h);
  ctx.filter = "none";
  ctx.globalCompositeOperation = "source-over";
}

/** The belt as a ring seen from above, front at the top. Wedge length shrinks as things get close. */
export function drawRing(canvas: HTMLCanvasElement, s: Safety | null) {
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  if (size < 20) return;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const c = size / 2;
  const rMax = c - 6;
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.lineWidth = 1;
  for (const f of [0.35, 0.7, 1]) {
    ctx.beginPath();
    ctx.arc(c, c, rMax * f, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let i = 0; i < 8; i++) {
    const d = s?.sectors[i] ?? null;
    const mid = (i * 45 - 90) * (Math.PI / 180);
    const a0 = mid - Math.PI / 8 + 0.03;
    const a1 = mid + Math.PI / 8 - 0.03;
    const reach = d === null ? 1 : Math.min(1, d / 3);
    ctx.beginPath();
    ctx.moveTo(c, c);
    ctx.arc(c, c, Math.max(6, rMax * reach), a0, a1);
    ctx.closePath();
    ctx.fillStyle = d === null ? "rgba(255,255,255,0.07)" : toneFor(d) + "66";
    ctx.fill();
    ctx.strokeStyle = d === null ? "rgba(255,255,255,0.22)" : toneFor(d);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(c, c, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

/** Eight cells under the video: metres and a bar per direction. */
export class BeltCells {
  private cells: HTMLElement[] = [];
  constructor(root: HTMLElement) {
    for (const name of SECTOR_NAMES) {
      const el = document.createElement("div");
      el.className = "cell";
      el.innerHTML = `<span>${name}</span><b>-</b><div class="bar"><i></i></div>`;
      root.appendChild(el);
      this.cells.push(el);
    }
  }
  update(s: Safety): void {
    s.sectors.forEach((d, i) => {
      const el = this.cells[i];
      el.className = `cell ${s.cues[i].mode === "off" ? "" : s.cues[i].mode}`;
      el.querySelector("b")!.textContent = d === null ? "clear" : `${d.toFixed(1)} m`;
      const fill = d === null ? 0 : Math.max(0.04, Math.min(1, 1 - d / 3));
      (el.querySelector("i") as HTMLElement).style.width = `${fill * 100}%`;
    });
  }
}
