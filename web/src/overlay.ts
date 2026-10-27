import { SECTOR_NAMES, type Safety } from "./types";

const COLORS = { ok: "#2e9e5b", warn: "#e0a030", near: "#d64545" };

function toneFor(d: number | null): string {
  if (d === null) return COLORS.ok;
  return d < 1 ? COLORS.near : d < 2 ? COLORS.warn : COLORS.ok;
}

/** Boxes with metres, drawn over the live video where the server saw them. */
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
  if (!s || !video.videoWidth) return;
  // The video is object-fit: contain, so find where it actually sits in the canvas.
  const scale = Math.min(cw / video.videoWidth, ch / video.videoHeight);
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  const ox = (cw - w) / 2;
  const oy = (ch - h) / 2;
  ctx.font = '500 13px "Atkinson Hyperlegible Next", system-ui, sans-serif';
  ctx.lineWidth = 2.5;
  for (const o of s.obstacles) {
    const [x0, y0, x1, y1] = o.box;
    const col = toneFor(o.dist);
    ctx.strokeStyle = col;
    ctx.strokeRect(ox + x0 * w, oy + y0 * h, (x1 - x0) * w, (y1 - y0) * h);
    const text = `${o.label}${o.dist === null ? "" : " " + o.dist.toFixed(1) + " m"}`;
    const tw = ctx.measureText(text).width + 10;
    const ty = Math.max(oy + 20, oy + y0 * h);
    ctx.fillStyle = col;
    ctx.fillRect(ox + x0 * w, ty - 20, tw, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, ox + x0 * w + 5, ty - 5);
  }
}

/** The belt as a ring seen from above, front at the top. Wedge length shrinks as things get close. */
export function drawRing(canvas: HTMLCanvasElement, s: Safety | null) {
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const c = size / 2;
  const rMax = c - 6;
  ctx.strokeStyle = "#dddddd";
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
    ctx.fillStyle = d === null ? "#f0f0ee" : toneFor(d) + "55";
    ctx.fill();
    ctx.strokeStyle = d === null ? "#dddddd" : toneFor(d);
    ctx.stroke();
  }
  ctx.fillStyle = "#161616";
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
