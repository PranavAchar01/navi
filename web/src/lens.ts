// What the Ray-Ban Display shows, worked out the way the phone app does it
// (ios/CameraAccess/MetaSight/GlassesDisplay.swift and WarningFilter in ServerLink.swift), so the
// judge view's lens mirror shows the card the glasses would show for the same safety message.

import type { Answer, Safety } from "./types";

/** clear < slow (1-2 m) < fast (0.5-1 m) < stop (under 0.5 m), as on the belt. */
export type Band = 0 | 1 | 2 | 3;
export const CLEAR = 0, SLOW = 1, FAST = 2, STOP = 3;

/** One palette for every surface: red is the stop zone only, amber is close, white is 1-2 m. */
export const BAND_RGB: Record<Band, [number, number, number]> = {
  0: [255, 255, 255],
  1: [255, 255, 255],
  2: [255, 190, 70],
  3: [255, 72, 72],
};
const GREEN: [number, number, number] = [110, 220, 140];

const DIRECTION = ["Ahead", "Ahead right", "Right", "Behind right", "Behind", "Behind left", "Left", "Ahead left"];

export interface Warning {
  sector: number;
  dist: number | null;
  band: Band;
}

function bandOfDistance(d: number | null): Band {
  if (d === null || !isFinite(d) || d >= 2) return CLEAR;
  return d >= 1 ? SLOW : d >= 0.5 ? FAST : STOP;
}

function bandOfMode(mode: string): Band {
  return mode === "slow" ? SLOW : mode === "fast" ? FAST : mode === "solid" ? STOP : CLEAR;
}

function outranks(a: Warning, b: Warning): boolean {
  if (a.band !== b.band) return a.band > b.band;
  if ((a.sector === 0) !== (b.sector === 0)) return a.sector === 0;
  return (a.dist ?? 0) < (b.dist ?? 0);
}

/** The most urgent obstacle the server actually cued, at the urgency it cued: what the belt feels.
 *  The front sector also counts the walking path, as the belt's planner does. */
export function cueWarning(s: Safety): Warning | null {
  const dists = s.sectors.slice(0, 8);
  if (s.path !== null) dists[0] = Math.min(dists[0] ?? Infinity, s.path);
  const modes = (s.cues ?? []).map((c) => c.mode as string);
  let best: Warning | null = null;
  for (let i = 0; i < 8; i++) {
    const mode = modes[i] ?? "unknown";
    if (mode === "unknown") continue;
    const cued = bandOfMode(mode);
    if (cued === CLEAR) continue;
    const d = dists[i] ?? null;
    const w: Warning = { sector: i, dist: d, band: Math.max(bandOfDistance(d), cued) as Band };
    if (!best || outranks(w, best)) best = w;
  }
  return best ? { ...best, band: bandOfMode(modes[best.sector]) } : null;
}

/** Distances in steps a glance can use: 0.1 m under 1 m, 0.5 m to 3 m, whole metres beyond. */
/** Metres as the lens says them: tenths under 1 m, halves to 3 m, whole metres beyond. */
export function roundDistance(d: number): number {
  const per = d < 1 ? 10 : d < 3 ? 2 : 1;
  return Math.max(1, Math.round(d * per)) / per;
}

export function distanceText(d: number | null): string {
  if (d === null) return "close";
  return `${roundDistance(d).toFixed(1)} m`;
}

/** "just now", "4 min ago", "1 h ago", rounded as the spoken answer is. */
export function ago(seenMs: number, nowMs: number): string {
  const s = Math.max(0, (nowMs - seenMs) / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(s / 3600);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

/** Hysteresis on the cued warning, the same holds as the phone: a zone shows only once the cue
 *  has held for its rise time, drops one zone at a time, and the number changes at most once a
 *  second. Without it the lens flickers between zones on every noisy frame. */
export class WarningFilter {
  shown: Warning | null = null;
  private heldSince = new Map<Band, number>();
  private belowSince: number | null = null;
  private numberAt = -Infinity;

  private static rise(b: Band): number {
    return b === STOP ? 0.15 : b === FAST ? 0.3 : 0.5;
  }
  private static fall(b: Band): number {
    return b === STOP ? 0.3 : 0.6;
  }

  update(raw: Warning | null, now: number): Warning | null {
    const slack = 0.001;
    const rawBand = raw?.band ?? CLEAR;
    for (const b of [SLOW, FAST, STOP] as Band[]) {
      if (rawBand >= b) this.heldSince.set(b, this.heldSince.get(b) ?? now);
      else this.heldSince.delete(b);
    }
    const ready = ([STOP, FAST, SLOW] as Band[]).find((b) => {
      const since = this.heldSince.get(b);
      return since !== undefined && now - since >= WarningFilter.rise(b) - slack;
    }) ?? CLEAR;
    let band: Band = this.shown?.band ?? CLEAR;
    if (ready > band) {
      band = ready;
      this.belowSince = null;
    } else if (rawBand < band) {
      const since = this.belowSince ?? now;
      this.belowSince = since;
      if (now - since >= WarningFilter.fall(band) - slack) {
        band = Math.max(ready, band - 1) as Band;
        this.belowSince = rawBand < band ? now : null;
      }
    } else {
      this.belowSince = null;
    }
    if (band === CLEAR) {
      this.shown = null;
      return null;
    }
    const zoneChanged = band !== this.shown?.band;
    if (raw && raw.band >= band && (zoneChanged || now - this.numberAt >= 1 - slack)) {
      if (zoneChanged || distanceText(raw.dist) !== distanceText(this.shown?.dist ?? null) || raw.sector !== this.shown?.sector) {
        this.numberAt = now;
      }
      this.shown = { sector: raw.sector, dist: raw.dist, band };
    } else if (this.shown) {
      this.shown = { ...this.shown, band };
    } else if (raw) {
      this.shown = { sector: raw.sector, dist: raw.dist, band };
      this.numberAt = now;
    }
    return this.shown;
  }
}

export type Glyph =
  | { kind: "obstacle"; bearing: number; band: Band }
  | { kind: "clear" }
  | { kind: "target"; bearing: number }
  | { kind: "pin" };

export interface Card {
  glyph: Glyph | null;
  title: string;
  detail: string | null;
  edge: Glyph | null;
}

export interface AnswerNote {
  text: string;
  label: string | null;
  near: string | null;
  seenMs: number | null;
  atMs: number;
}

export type Guide = Safety["guide"];

export const ANSWER_SECONDS = 12;

export function answerNote(a: Answer, nowMs: number): AnswerNote {
  return { text: a.text, label: a.label, near: a.pin?.near?.[0] ?? null, seenMs: a.pin ? a.pin.ts * 1000 : null, atMs: nowMs };
}

function clipped(text: string, limit = 48): string {
  if (text.length <= limit) return text;
  const head = text.slice(0, limit - 1);
  const cut = head.lastIndexOf(" ") > 0 ? head.slice(0, head.lastIndexOf(" ")) : head;
  return cut.replace(/[\s.,;:!?]+$/, "") + "…";
}

function directionWord(bearing: number): string {
  const sector = ((Math.round(bearing / 45) % 8) + 8) % 8;
  return DIRECTION[sector];
}

function lastSeen(a: AnswerNote | null, nowMs: number): string | null {
  const parts = [a?.near ?? null, a?.seenMs != null ? ago(a.seenMs, nowMs) : null].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

function obstacleCard(w: Warning): Card {
  const glyph: Glyph = { kind: "obstacle", bearing: w.sector * 45, band: w.band };
  if (w.band === STOP) return { glyph, title: "Stop", detail: DIRECTION[w.sector], edge: null };
  return { glyph, title: distanceText(w.dist), detail: DIRECTION[w.sector], edge: null };
}

function guiding(g: NonNullable<Guide>, answer: AnswerNote | null, nowMs: number): Card {
  const note = answer?.label === g.label ? answer : null;
  if (g.dist !== null && g.bearing !== null) {
    const near = note?.near ? `${g.label} · ${note.near}` : null;
    return { glyph: { kind: "target", bearing: g.bearing }, title: distanceText(g.dist), detail: near && near.length <= 22 ? near : g.label, edge: null };
  }
  if (g.bearing !== null) return { glyph: { kind: "target", bearing: g.bearing }, title: g.label, detail: directionWord(g.bearing), edge: null };
  if (g.dist !== null) return { glyph: { kind: "pin" }, title: g.label, detail: distanceText(g.dist), edge: null };
  return { glyph: { kind: "pin" }, title: g.label, detail: lastSeen(note, nowMs), edge: null };
}

/** Most urgent first: stop, then a guide with a live fix, then a fresh answer, then any cued
 *  obstacle, then the pin of an item still being looked for. Mirrors DisplayCard.make. */
export function makeCard(warning: Warning | null, guide: Guide, answer: AnswerNote | null, nowMs: number): Card {
  if (warning && warning.band === STOP) return obstacleCard(warning);
  const edge: Glyph | null = warning ? { kind: "obstacle", bearing: warning.sector * 45, band: warning.band } : null;
  const edged = (c: Card): Card => ({ ...c, edge });
  if (guide && (guide.dist !== null || guide.bearing !== null)) return edged(guiding(guide, answer, nowMs));
  if (answer && nowMs - answer.atMs < ANSWER_SECONDS * 1000) {
    if (guide && guide.label === answer.label) return edged(guiding(guide, answer, nowMs));
    if (answer.label) return edged({ glyph: { kind: "pin" }, title: answer.label, detail: lastSeen(answer, nowMs) ?? clipped(answer.text), edge: null });
    return edged({ glyph: null, title: clipped(answer.text), detail: null, edge: null });
  }
  if (warning) return obstacleCard(warning);
  if (guide) return guiding(guide, answer, nowMs);
  return { glyph: { kind: "clear" }, title: "Clear", detail: null, edge: null };
}

function sameCard(a: Card, b: Card): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** The lens refresh: obstacle cards go up at once, other cards wait out a 1 s dwell so the lens
 *  does not flicker. Call every 100 ms like the phone does. */
export class LensState {
  card: Card = { glyph: { kind: "clear" }, title: "Clear", detail: null, edge: null };
  private shownAt = -Infinity;

  refresh(next: Card, nowMs: number): Card {
    if (sameCard(next, this.card)) return this.card;
    const obstacle = next.glyph?.kind === "obstacle";
    if (!obstacle && nowMs - this.shownAt < 1000) return this.card;
    this.card = next;
    this.shownAt = nowMs;
    return this.card;
  }
}

function rgb(c: [number, number, number], a = 1): string {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

function fitted(ctx: CanvasRenderingContext2D, text: string, weight: number, size: number, width: number): string {
  ctx.font = `${weight} ${size}px -apple-system, "SF Pro Display", system-ui, sans-serif`;
  const w = ctx.measureText(text).width;
  const s = w > width ? Math.floor((size * width) / w) : size;
  return `${weight} ${s}px -apple-system, "SF Pro Display", system-ui, sans-serif`;
}

/** The card as the glasses draw it: one 528 px square on black (black is clear glass). */
export function drawCard(canvas: HTMLCanvasElement, card: Card): void {
  const side = 528;
  if (canvas.width !== side) {
    canvas.width = side;
    canvas.height = side;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, side, side);
  const g = card.glyph;
  const color = g?.kind === "obstacle" ? BAND_RGB[g.band] : g?.kind === "clear" ? GREEN : ([255, 255, 255] as [number, number, number]);
  if (card.edge?.kind === "obstacle") {
    // A straight bar on the nearest side, never across a corner.
    const sector = Math.round(card.edge.bearing / 45) % 8;
    const deg = [0, 90, 90, 90, 180, 270, 270, 270][sector];
    ctx.save();
    ctx.translate(side / 2, side / 2);
    ctx.rotate((deg * Math.PI) / 180);
    ctx.fillStyle = rgb(BAND_RGB[card.edge.band]);
    ctx.beginPath();
    ctx.roundRect(-side * 0.22, -side / 2 + 6, side * 0.44, 18, 9);
    ctx.fill();
    ctx.restore();
  }
  const maxW = side - 16;
  const titleLines = g ? [card.title] : wrap(ctx, card.title, maxW, "700 54px -apple-system, system-ui, sans-serif").slice(0, 3);
  const titleFont = g ? fitted(ctx, card.title, 800, 112, maxW) : "700 54px -apple-system, system-ui, sans-serif";
  const titleSize = parseInt(titleFont.split(" ")[1]);
  const lineH = titleSize * 1.2;
  const titleH = lineH * titleLines.length;
  const detailFont = card.detail ? fitted(ctx, card.detail, 600, 46, maxW) : "";
  const detailH = card.detail ? parseInt(detailFont.split(" ")[1]) * 1.2 : 0;
  const glyphH = g ? 176 : 0;
  const gap = 14;
  const total = glyphH + (glyphH ? gap : 0) + titleH + (card.detail ? gap + detailH : 0);
  let y = (side - total) / 2;
  if (g) {
    drawGlyph(ctx, g, rgb(color), side / 2, y + glyphH / 2);
    y += glyphH + gap;
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = titleFont;
  ctx.fillStyle = rgb(color);
  titleLines.forEach((line, i) => ctx.fillText(line, side / 2, y + lineH * (i + 0.5)));
  y += titleH + gap;
  if (card.detail) {
    ctx.font = detailFont;
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(card.detail, side / 2, y + detailH / 2);
  }
}

function wrap(ctx: CanvasRenderingContext2D, text: string, width: number, font: string): string[] {
  ctx.font = font;
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width > width && line) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

function drawGlyph(ctx: CanvasRenderingContext2D, g: Glyph, color: string, cx: number, cy: number): void {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  if (g.kind === "pin") {
    ctx.beginPath();
    ctx.arc(0, -30, 56, Math.PI * 0.8, Math.PI * 0.2, false);
    ctx.lineTo(0, 88);
    ctx.closePath();
    ctx.moveTo(21, -30);
    ctx.arc(0, -30, 21, 0, Math.PI * 2, true);
    ctx.fill("evenodd");
  } else if (g.kind === "clear") {
    ctx.lineWidth = 26;
    ctx.beginPath();
    ctx.arc(0, 0, 66, 0, Math.PI * 2);
    ctx.stroke();
  } else if (g.kind === "obstacle" && g.band === STOP) {
    ctx.beginPath();
    for (let k = 0; k < 8; k++) {
      const a = ((k + 0.5) * Math.PI) / 4;
      ctx.lineTo(86 * Math.cos(a), 86 * Math.sin(a));
    }
    ctx.closePath();
    ctx.fill();
  } else if (g.kind === "obstacle") {
    ctx.rotate((g.bearing * Math.PI) / 180);
    ctx.beginPath();
    ctx.arc(0, 50, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.roundRect(-80, -72, 160, 34, 8);
    ctx.fill();
  } else if (g.kind === "target") {
    ctx.rotate((g.bearing * Math.PI) / 180);
    ctx.beginPath();
    for (const [x, y] of [[0, -86], [68, -8], [26, -8], [26, 82], [-26, 82], [-26, -8], [-68, -8]]) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}
