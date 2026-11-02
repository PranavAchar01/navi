import { describe, expect, it } from "vitest";
import { type Cell, angles, heat, spread } from "./beltlayout";

const AMBER: [number, number, number] = [255, 190, 70];
const off = (): Cell[] => Array.from({ length: 8 }, () => ({ rgb: null, level: 0 }));

describe("spread", () => {
  it("folds the front cell onto both front motors at 1/sqrt(2), like the firmware", () => {
    const c = off();
    c[0] = { rgb: AMBER, level: 1 };
    const m = spread(c, 4);
    expect(m[0].level).toBeCloseTo(Math.SQRT1_2);
    expect(m[3].level).toBeCloseTo(Math.SQRT1_2);
    expect(m[1].level).toBe(0);
    expect(m[2].level).toBe(0);
    expect(m[0].rgb).toEqual(AMBER);
  });

  it("drives only its own motor for a diagonal cell", () => {
    const c = off();
    c[3] = { rgb: AMBER, level: 1 };
    expect(spread(c, 4).map((x) => x.level)).toEqual([0, 1, 0, 0]);
  });

  it("splits a cue straight ahead evenly over the two front motors at constant power", () => {
    const c = off();
    c[0] = { rgb: AMBER, level: 1 };
    const m = spread(c, 8).map((x) => x.level);
    expect(m[0]).toBeCloseTo(Math.SQRT1_2);
    expect(m[7]).toBeCloseTo(Math.SQRT1_2);
    expect(m.filter((v) => v > 0)).toHaveLength(2);
  });

  it("lands a cue behind on the two back motors, never the front", () => {
    const c = off();
    c[4] = { rgb: AMBER, level: 1 };
    const m = spread(c, 8).map((x) => x.level);
    expect(m[3]).toBeCloseTo(m[4]);
    expect(m[3]).toBeGreaterThan(0.5);
    expect(m[0] + m[7]).toBe(0);
  });

  it("runs a few neighbouring motors on 16 for one cue", () => {
    const c = off();
    c[2] = { rgb: AMBER, level: 1 };
    const lit = spread(c, 16).filter((x) => x.level > 0);
    expect(lit.length).toBeGreaterThanOrEqual(2);
    expect(lit.length).toBeLessThanOrEqual(3);
  });

  it("puts four motors across the front and leaves the middle for the board", () => {
    expect(angles(4)).toEqual([30, 75, -75, -30]);
    expect(angles(8)).toEqual([30, 75, 105, 150, -150, -105, -75, -30]);
    expect(Math.min(...angles(16).map(Math.abs))).toBeGreaterThan(25);
  });
});

describe("heat", () => {
  it("goes from yellow at a light buzz to red at full power", () => {
    const [, gLow] = heat(0.1);
    const [, gHigh] = heat(1);
    expect(gLow).toBeGreaterThan(150);
    expect(gHigh).toBeLessThan(60);
  });
});
