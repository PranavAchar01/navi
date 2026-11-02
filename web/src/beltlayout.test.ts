import { describe, expect, it } from "vitest";
import { type Cell, angles, spread } from "./beltlayout";

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

  it("lights a between motor only when both neighbours run", () => {
    const c = off();
    c[1] = { rgb: AMBER, level: 1 };
    expect(spread(c, 16).filter((x) => x.level > 0)).toHaveLength(1);
    c[2] = { rgb: AMBER, level: 0.5 };
    const m = spread(c, 16);
    expect(m[2].level).toBe(1);
    expect(m[3].level).toBe(0.5);
    expect(m[4].level).toBe(0.5);
  });

  it("puts the motors clockwise from the front", () => {
    expect(angles(4)).toEqual([45, 135, 225, 315]);
    expect(angles(16)[4]).toBe(90);
  });
});
