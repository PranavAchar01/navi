import { describe, expect, it } from "vitest";
import { type NaviInputs, gazeOf, lineFor, moodOf } from "./navimood";

const calm: NaviInputs = {
  online: true,
  listening: false,
  waiting: false,
  alert: false,
  ttc: null,
  path: null,
  bearing: null,
  sinceFoundMs: null,
  quietMs: 0,
};

describe("moodOf", () => {
  it("rests idle and dozes off after a quiet minute", () => {
    expect(moodOf(calm)).toBe("idle");
    expect(moodOf({ ...calm, quietMs: 61_000 })).toBe("sleepy");
  });

  it("puts safety first: an obstacle beats listening, guiding and a fresh find", () => {
    const busy = { ...calm, listening: true, bearing: 40, sinceFoundMs: 100 };
    expect(moodOf({ ...busy, ttc: 1.2 })).toBe("careful");
    expect(moodOf({ ...busy, path: 0.5 })).toBe("careful");
    expect(moodOf({ ...busy, ttc: 3, path: 2 })).toBe("listening");
  });

  it("goes offline before anything else, and worries over a check-in", () => {
    expect(moodOf({ ...calm, online: false, alert: true })).toBe("offline");
    expect(moodOf({ ...calm, alert: true, ttc: 0.5 })).toBe("worried");
  });

  it("thinks while a question is out, beams on a find, then guides", () => {
    expect(moodOf({ ...calm, waiting: true })).toBe("thinking");
    expect(moodOf({ ...calm, bearing: 30, sinceFoundMs: 1000 })).toBe("happy");
    expect(moodOf({ ...calm, bearing: 30, sinceFoundMs: 9000 })).toBe("guiding");
  });
});

describe("gazeOf", () => {
  it("looks toward the bearing, and over a shoulder for things behind", () => {
    expect(gazeOf("guiding", 90).x).toBeCloseTo(1);
    expect(gazeOf("guiding", -90).x).toBeCloseTo(-1);
    expect(gazeOf("guiding", 0).x).toBeCloseTo(0);
    expect(gazeOf("guiding", 170)).toEqual({ x: 1, y: 0.35 });
    expect(gazeOf("guiding", 370).x).toBeCloseTo(Math.sin(Math.PI / 18));
  });

  it("looks up at you while listening, whatever it was guiding to", () => {
    expect(gazeOf("listening", 90)).toEqual({ x: 0, y: -0.7 });
  });
});

describe("lineFor", () => {
  it("keeps lines short and names the thing", () => {
    expect(lineFor("found", "keys")).toBe("There it is, your keys!");
    expect(lineFor("answer", "a very long label ".repeat(6)).length).toBeLessThanOrEqual(64);
    expect(lineFor("hello")).not.toMatch(/—/);
  });
});
