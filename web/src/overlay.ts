import { type Belt, type Hazard, SECTOR_NAMES, type Safety } from "./types";

// Hazards: only the obstacles in the walking path, nearest first, each a smooth thin contour
// with a soft inner fade and a small distance tag. Amber when you are approaching (under 1.4 m),
// red when you are too close (under 0.9 m). Hysteresis on the level, easing on colour and alpha.
const AMBER: [number, number, number] = [255, 190, 70];
const RED: [number, number, number] = [255, 72, 72];
interface Look { level: number; rgb: [number, number, number]; alpha: number; seen: number; x: number; y: number }
let looks: Look[] = [];

function look(h: Hazard, now: number): Look {
  let best: Look | null = null;
  let bd = 0.15;
  for (const l of looks) {
    const dd = Math.hypot(l.x - h.tag[0], l.y - h.tag[1]);
    if (dd < bd && l.seen !== now) {
      best = l;
      bd = dd;
    }
  }
  if (!best) {
    best = { level: 1, rgb: [...AMBER], alpha: 0, seen: now, x: h.tag[0], y: h.tag[1] };
    looks.push(best);
  }
  best.x += (h.tag[0] - best.x) * 0.5;
  best.y += (h.tag[1] - best.y) * 0.5;
  best.seen = now;
  if (best.level === 1 && h.dist < 0.9) best.level = 2;
  else if (best.level === 2 && h.dist > 1.05) best.level = 1;
  const target = best.level === 2 ? RED : AMBER;
  for (let i = 0; i < 3; i++) best.rgb[i] += (target[i] - best.rgb[i]) * 0.2;
  best.alpha += (1 - best.alpha) * 0.25;
  return best;
}

function smoothPath(ctx: CanvasRenderingContext2D, p: number[], ox: number, oy: number, w: number, h: number) {
  const n = p.length / 2;
  const X = (i: number) => ox + p[(i % n) * 2] * w;
  const Y = (i: number) => oy + p[(i % n) * 2 + 1] * h;
  ctx.beginPath();
  ctx.moveTo((X(0) + X(1)) / 2, (Y(0) + Y(1)) / 2);
  for (let i = 1; i <= n; i++) ctx.quadraticCurveTo(X(i), Y(i), (X(i) + X(i + 1)) / 2, (Y(i) + Y(i + 1)) / 2);
  ctx.closePath();
}

// The contour without the stretches that run along the frame edge: the edge of the picture is
// not the edge of the object, and a line there reads as a border, not an obstacle.
function edgePath(ctx: CanvasRenderingContext2D, p: number[], ox: number, oy: number, w: number, h: number) {
  const n = p.length / 2;
  const px = (i: number) => p[(i % n) * 2];
  const py = (i: number) => p[(i % n) * 2 + 1];
  const onEdge = (i: number) => px(i) < 0.006 || px(i) > 0.994 || py(i) < 0.006 || py(i) > 0.994;
  const X = (i: number) => ox + px(i) * w;
  const Y = (i: number) => oy + py(i) * h;
  ctx.beginPath();
  let pen = false;
  for (let i = 1; i <= n; i++) {
    if (onEdge(i)) {
      pen = false;
      continue;
    }
    if (!pen) ctx.moveTo((X(i - 1) + X(i)) / 2, (Y(i - 1) + Y(i)) / 2);
    ctx.quadraticCurveTo(X(i), Y(i), (X(i) + X(i + 1)) / 2, (Y(i) + Y(i + 1)) / 2);
    pen = true;
  }
}

function drawHazards(ctx: CanvasRenderingContext2D, hz: Hazard[], ox: number, oy: number, w: number, h: number) {
  const now = performance.now();
  const shown = hz.map((x) => [x, look(x, now)] as const);
  looks = looks.filter((l) => now - l.seen < 400);
  for (const [x, l] of [...shown].reverse()) {
    if (x.poly.length < 6) continue;
    const [r, g, b] = l.rgb.map(Math.round);
    const red = l.level === 2;
    const beat = red ? 0.75 + 0.25 * Math.sin(now / 130) : 1;
    smoothPath(ctx, x.poly, ox, oy, w, h);
    // Inner fade: clip to the shape and stroke wide and soft, so it is brightest at the edge.
    ctx.save();
    ctx.clip();
    ctx.shadowColor = `rgba(${r},${g},${b},${(0.55 * l.alpha * beat).toFixed(3)})`;
    ctx.shadowBlur = w / 45;
    ctx.strokeStyle = `rgba(${r},${g},${b},${(0.22 * l.alpha * beat).toFixed(3)})`;
    ctx.lineWidth = w / 70;
    ctx.stroke();
    ctx.fillStyle = `rgba(${r},${g},${b},${((red ? 0.1 : 0.05) * l.alpha).toFixed(3)})`;
    ctx.fill();
    ctx.restore();
    // Crisp contour.
    edgePath(ctx, x.poly, ox, oy, w, h);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.shadowColor = `rgba(${r},${g},${b},0.9)`;
    ctx.shadowBlur = 10;
    ctx.strokeStyle = `rgba(${r},${g},${b},${(0.95 * l.alpha).toFixed(3)})`;
    ctx.lineWidth = red ? 2.6 : 1.8;
    ctx.stroke();
    ctx.shadowBlur = 0;
    // Distance tag just above the part that is in the way.
    const tx = ox + x.tag[0] * w;
    const ty = Math.max(oy + 26, oy + x.tag[1] * h - 14);
    const text = `${x.dist.toFixed(1)} m`;
    ctx.font = '600 13px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
    const tw = ctx.measureText(text).width + 26;
    ctx.fillStyle = `rgba(10,14,30,${(0.62 * l.alpha).toFixed(3)})`;
    ctx.beginPath();
    ctx.roundRect(tx - tw / 2, ty - 13, tw, 24, 12);
    ctx.fill();
    ctx.strokeStyle = `rgba(${r},${g},${b},${(0.7 * l.alpha).toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(tx - tw / 2 + 11, ty - 1, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = `rgba(255,255,255,${l.alpha.toFixed(3)})`;
    ctx.fillText(text, tx - tw / 2 + 19, ty + 4);
  }
}

/** The in-path obstacles over the live video. */
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
  if (!s || !video.videoWidth || !s.hazards) return;
  // The video is object-fit: contain, so find where it actually sits in the canvas.
  const scale = Math.min(cw / video.videoWidth, ch / video.videoHeight);
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  drawHazards(ctx, s.hazards, (cw - w) / 2, (ch - h) / 2, w, h);
}

const MODE_RGB: Record<string, [number, number, number]> = {
  slow: [255, 190, 70],
  fast: [255, 72, 72],
  solid: [255, 72, 72],
  lost: [170, 120, 255],
  guide: [90, 200, 255],
};

/** The belt seen from above, front at the top: the wearer, the strap, and the eight numbered
 *  motors. Each motor is drawn from the belt's own telemetry, so it lights and shakes exactly
 *  when the firmware is driving it. */
export function drawRing(canvas: HTMLCanvasElement, b: Belt | null, s: Safety | null) {
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
  const R = c - 24;
  const now = performance.now();
  // Wearer from above: shoulders and head, nose pointing forward.
  ctx.fillStyle = "rgba(255,255,255,0.10)";
  ctx.beginPath();
  ctx.ellipse(c, c, R * 0.62, R * 0.4, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.22)";
  ctx.beginPath();
  ctx.arc(c, c, R * 0.22, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(c - 5, c - R * 0.2);
  ctx.lineTo(c, c - R * 0.32);
  ctx.lineTo(c + 5, c - R * 0.2);
  ctx.fill();
  // Guiding to a remembered item: a dotted line from the wearer toward it.
  const g = s?.guide;
  if (g && g.bearing !== null) {
    const a = ((g.bearing - 90) * Math.PI) / 180;
    ctx.strokeStyle = "rgba(90,200,255,0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(c + R * 0.25 * Math.cos(a), c + R * 0.25 * Math.sin(a));
    ctx.lineTo(c + (R + 20) * Math.cos(a), c + (R + 20) * Math.sin(a));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // The strap.
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(c, c, R, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = '600 9px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
  ctx.textAlign = "center";
  ctx.fillText(b?.lost ? "BELT LINK LOST" : b?.found ? "FOUND IT" : "BELT", c, c + R * 0.62);
  ctx.font = '600 10px "Atkinson Hyperlegible Mono", ui-monospace, monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < 8; i++) {
    const a = (i * 45 - 90) * (Math.PI / 180);
    const mode = !b ? "off" : b.lost ? "lost" : b.modes[i];
    const duty = b ? b.duty[i] / 255 : 0;
    const rgb = MODE_RGB[mode];
    // A running motor shakes by a pixel or two, so the vibration is visible in a still frame too.
    const shake = duty * 1.8;
    const x = c + R * Math.cos(a) + shake * Math.sin(now / 9 + i);
    const y = c + R * Math.sin(a) + shake * Math.cos(now / 11 + i);
    if (rgb && duty > 0) {
      const [r, g, bl] = rgb;
      for (const k of [1, 2]) {
        ctx.strokeStyle = `rgba(${r},${g},${bl},${(0.5 * duty / k).toFixed(3)})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 11 + 6 * k * duty, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.shadowColor = `rgba(${r},${g},${bl},0.9)`;
      ctx.shadowBlur = 14 * duty;
    }
    const lit = rgb ? `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.35 + 0.65 * duty).toFixed(3)})` : "rgba(40,48,80,0.95)";
    ctx.fillStyle = lit;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = mode === "unknown" ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.6)";
    ctx.setLineDash(mode === "unknown" ? [2, 2] : []);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    ctx.fillText(String(i + 1), x, y + 0.5);
  }
}

/** Eight cells under the video: metres and a bar per direction. */
export class BeltCells {
  private cells: HTMLElement[] = [];
  constructor(private root: HTMLElement) {
    for (const name of SECTOR_NAMES) {
      const el = document.createElement("div");
      el.className = "cell";
      el.innerHTML = `<span><em>${SECTOR_NAMES.indexOf(name) + 1}</em>${name.replace("-right", "-R").replace("-left", "-L")}</span><b>-</b><div class="bar"><i></i></div><em class="motor"></em>`;
      root.appendChild(el);
      this.cells.push(el);
    }
  }
  /** Distances, from the safety loop. */
  update(s: Safety): void {
    s.sectors.forEach((sd, i) => {
      const el = this.cells[i];
      // The front motor answers to the walking path too, so show the same number it acts on.
      const d = i === 0 && s.path !== null ? Math.min(sd ?? Infinity, s.path) : sd;
      el.querySelector("b")!.textContent = !s.known[i] ? "no data" : d === null ? "clear" : `${d.toFixed(1)} m`;
      const fill = d === null || !s.known[i] ? 0 : Math.max(0.04, Math.min(1, 1 - d / 3));
      (el.querySelector("i") as HTMLElement).style.width = `${fill * 100}%`;
    });
  }
  /** What each motor is doing right now, from the belt's own telemetry. */
  motors(b: Belt): void {
    this.root.classList.toggle("lost", b.lost);
    b.duty.forEach((v, i) => {
      const el = this.cells[i];
      const mode = b.lost ? "lost" : b.modes[i];
      el.className = `cell ${mode === "off" ? "" : mode}`;
      el.style.setProperty("--duty", (v / 255).toFixed(2));
    });
  }
}
