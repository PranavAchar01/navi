// Contour easing: outlines from successive safety messages blend point by point.
import { describe, expect, it } from "vitest";
import { resample, rimRuns } from "./overlay";

const square = [0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6];

describe("resample", () => {
  it("gives 64 points on the outline whatever the input count", () => {
    const r = resample(square, null);
    expect(r.length).toBe(128);
    for (let i = 0; i < r.length; i += 2) {
      const onX = Math.abs(r[i] - 0.2) < 1e-9 || Math.abs(r[i] - 0.6) < 1e-9;
      const onY = Math.abs(r[i + 1] - 0.2) < 1e-9 || Math.abs(r[i + 1] - 0.6) < 1e-9;
      expect(onX || onY).toBe(true);
    }
  });
  it("lines a reversed, rotated copy up with the previous outline", () => {
    const prev = resample(square, null);
    const turned = [0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.6]; // other winding, other start
    const r = resample(turned, prev);
    const err = r.reduce((s, v, i) => s + Math.abs(v - prev[i]), 0) / r.length;
    expect(err).toBeLessThan(1e-9);
  });
});

describe("rimRuns", () => {
  const W = 560;
  const H = 1080;
  // A blob in the middle of the picture: the whole rim at full strength.
  const circle = resample(
    [...Array(64).keys()].flatMap((i) => [0.5 + 0.2 * Math.cos((i / 64) * 2 * Math.PI), 0.5 + 0.1 * Math.sin((i / 64) * 2 * Math.PI)]),
    null,
  );
  it("a free blob keeps its whole rim", () => {
    const runs = rimRuns(circle, W, H);
    expect(runs).toHaveLength(1);
    expect(runs[0].a).toBe(1);
  });
  it("no rim on the picture border or a long straight cut, fading toward them", () => {
    // Rectangle from the bottom edge up to a straight cut across the width at y = 0.3.
    const rect = resample([0.05, 0.3, 0.95, 0.3, 0.95, 1, 0.05, 1], null);
    const runs = rimRuns(rect, W, H);
    const at = (x: number, y: number) =>
      Math.max(0, ...runs.filter((r) => r.idx.some((i) => Math.hypot(rect[i * 2] - x, rect[i * 2 + 1] - y) < 0.02)).map((r) => r.a));
    expect(at(0.5, 0.3)).toBe(0); // the cut
    expect(at(0.5, 1)).toBe(0); // the bottom border
    expect(at(0.05, 0.65)).toBe(1); // middle of a side
    expect(runs.some((r) => r.a > 0 && r.a < 1)).toBe(true); // a fade, not a hard stop
  });
});
