// The lens mirror must make the same calls as the phone; these are the phone's own cases
// (ios/CameraAccessTests/WarningFilterTests.swift and DisplayCardTests.swift).
import { describe, expect, it } from "vitest";
import { FAST, SLOW, STOP, WarningFilter, cueWarning, distanceText, makeCard, type Band } from "./lens";
import type { Safety } from "./types";

const t0 = 1000;
const w = (band: Band, dist: number, sector = 0) => ({ sector, dist, band });

describe("WarningFilter", () => {
  it("ignores a blip", () => {
    const f = new WarningFilter();
    f.update(w(FAST, 0.8), t0);
    f.update(w(FAST, 0.8), t0 + 0.2);
    expect(f.update(null, t0 + 0.25)).toBeNull();
  });
  it("rises on a held cue, and stop is quick", () => {
    const f = new WarningFilter();
    f.update(w(FAST, 0.8), t0);
    expect(f.update(w(FAST, 0.8), t0 + 0.3)?.band).toBe(FAST);
    f.update(w(STOP, 0.4), t0 + 0.35);
    expect(f.update(w(STOP, 0.4), t0 + 0.5)?.band).toBe(STOP);
  });
  it("falls one zone at a time", () => {
    const f = new WarningFilter();
    f.update(w(STOP, 0.3), t0);
    f.update(w(STOP, 0.3), t0 + 0.2);
    expect(f.update(null, t0 + 0.3)?.band).toBe(STOP);
    expect(f.update(null, t0 + 0.6)?.band).toBe(FAST);
    expect(f.update(null, t0 + 1.2)?.band).toBe(SLOW);
    expect(f.update(null, t0 + 1.8)).toBeNull();
  });
  it("moves the number at most once a second", () => {
    const f = new WarningFilter();
    f.update(w(FAST, 0.8), t0);
    expect(f.update(w(FAST, 0.8), t0 + 0.3)?.dist).toBe(0.8);
    expect(f.update(w(FAST, 0.7), t0 + 0.6)?.dist).toBe(0.8);
    expect(f.update(w(FAST, 0.7), t0 + 1.3)?.dist).toBe(0.7);
  });
});

describe("cards", () => {
  it("rounds distances the way a glance reads them", () => {
    expect(distanceText(0.84)).toBe("0.8 m");
    expect(distanceText(1.3)).toBe("1.5 m");
    expect(distanceText(null)).toBe("close");
  });
  it("takes only cued sectors, at the cued urgency, with the path in front", () => {
    const s = {
      sectors: [1.6, null, 0.4, null, null, null, null, null],
      path: 0.8,
      cues: ["fast", "off", "off", "off", "off", "off", "off", "off"].map((mode) => ({ mode })),
    } as unknown as Safety;
    expect(cueWarning(s)).toEqual({ sector: 0, dist: 0.8, band: FAST });
  });
  it("lets only stop take the lens from an answer, which keeps the warning as an edge", () => {
    const answer = { text: "On the desk", label: "laptop", near: "desk", seenMs: 0, atMs: 10_000 };
    const fast = makeCard(w(FAST, 0.7, 2), null, answer, 11_000);
    expect(fast.title).toBe("laptop");
    expect(fast.edge).toEqual({ kind: "obstacle", bearing: 90, band: FAST });
    expect(makeCard(w(STOP, 0.3), null, answer, 11_000).title).toBe("Stop");
    expect(makeCard(null, null, answer, 23_000).title).toBe("Clear");
  });
  it("points at the item when the server has a fix", () => {
    const card = makeCard(null, { label: "laptop", dist: 2.1, bearing: -40 }, null, 0);
    expect(card.glyph).toEqual({ kind: "target", bearing: -40 });
    expect(card.title).toBe("2.0 m");
  });
});
