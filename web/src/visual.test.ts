// Guide sign: the arc, the ring segment and the minimap ribbon all come from one bearing.
import { describe, expect, it } from "vitest";
import { bearingDeg, guideBearing, isPerson, motorAt, peopleBearings } from "./visual";

type V = [number, number, number];
const add = (a: V, b: V, k: number): V => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// The server's frames: y-up map with its yaw = atan2(fwd_x, fwd_z) (heading.py), the harness's
// three.js pose facing -z, and an OpenCV-style map (x right, y down, z forward, up = -y).
const frames: { name: string; pos: V; fwd: V; up: V }[] = [
  { name: "y-up, yaw 0", pos: [0, 1.5, 0], fwd: [0, 0, 1], up: [0, 1, 0] },
  { name: "y-up, facing -z", pos: [0, 1.5, 1.5], fwd: [0, 0, -1], up: [0, 1, 0] },
  { name: "y-up, yaw 0.7", pos: [2, 1.5, -1], fwd: [Math.sin(0.7), 0, Math.cos(0.7)], up: [0, 1, 0] },
  { name: "OpenCV, y down", pos: [0, -1.5, 0], fwd: [0, 0, 1], up: [0, -1, 0] },
];

describe("bearingDeg", () => {
  for (const f of frames) {
    const right = cross(f.fwd, f.up);
    const at = (dir: V) => bearingDeg(f.pos, f.fwd, f.up, add(add(f.pos, dir, 2), f.up, -0.8));
    it(`${f.name}: ahead is 0, no arc`, () => {
      expect(Math.abs(at(f.fwd))).toBeLessThan(12);
      expect(motorAt(at(f.fwd))).toBe(0);
    });
    it(`${f.name}: right is +90, right arc, motor index 2 (third motor)`, () => {
      expect(at(right)).toBeCloseTo(90, 6);
      expect(motorAt(at(right))).toBe(2);
    });
    it(`${f.name}: left is -90, left arc, motor index 6 (seventh motor)`, () => {
      expect(at([-right[0], -right[1], -right[2]])).toBeCloseTo(-90, 6);
      expect(motorAt(at([-right[0], -right[1], -right[2]]))).toBe(6);
    });
  }
  it("matches heading.py: yaw 0 in a y-up map, the wearer's right is -x", () => {
    expect(bearingDeg([0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0])).toBeCloseTo(90, 6);
  });
  it("OpenCV: x is right", () => {
    expect(bearingDeg([0, 0, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0])).toBeCloseTo(90, 6);
  });
});

describe("server guide and people", () => {
  const g = { label: "keys", dist: 2, bearing: 30 };
  it("prefers the server's bearing_deg", () => {
    expect(guideBearing({ ...g, pin_id: 4, bearing_deg: -40, dist_m: 2, turn: "left" }, 22)).toBe(-40);
  });
  it("falls back to the client's bearing without bearing_deg", () => {
    expect(guideBearing(g, 22)).toBe(22);
    expect(guideBearing({ ...g, bearing_deg: null }, 22)).toBe(22);
    expect(guideBearing(null, null)).toBeNull();
  });
  it("maps people bearing_deg to bearingDeg and drops bad ones", () => {
    const m = { type: "people" as const, people: [{ bearing_deg: -70 }, { bearing_deg: 15.5 }, { bearing_deg: NaN }] };
    expect(peopleBearings(m)).toEqual([{ bearingDeg: -70 }, { bearingDeg: 15.5 }]);
    expect(peopleBearings({ type: "people", people: [] })).toEqual([]);
  });
});

describe("isPerson", () => {
  it("hides people classes, keeps objects", () => {
    expect(isPerson("person")).toBe(true);
    expect(isPerson(" Person ")).toBe(true);
    expect(isPerson("keys")).toBe(false);
    expect(isPerson("personal laptop")).toBe(false);
  });
});
