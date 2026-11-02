/** How the belt's eight protocol cells land on a belt of 4, 8 or 16 motors, for the 3D belt.
 *  Pure, so the tests run without WebGL. */

export type Layout = 4 | 8 | 16;
export const LAYOUTS: Layout[] = [4, 8, 16];

/** One motor's look: its colour (null = dark) and how hard it runs right now, 0..1. */
export interface Cell {
  rgb: [number, number, number] | null;
  level: number;
}

/** Where each motor sits, degrees clockwise from the front (negative = the wearer's left).
 *  The middle of the front holds the ESP32, so no motor sits at 0.
 *  - 4: the belt as built, four across the front. In firmware order D26, D14, D13, D4: the
 *    firmware's front-right and back-right are the near and far right motors.
 *  - 8: those four plus four across the back.
 *  - 16: eight a side, evenly from the board to the buckle. */
export function angles(n: Layout): number[] {
  if (n === 4) return [30, 75, -75, -30];
  const right = n === 8 ? [30, 75, 105, 150] : Array.from({ length: 8 }, (_, i) => 27 + (143 * i) / 7);
  return [...right, ...right.map((a) => -a).reverse()];
}

/** GPIO of each motor on the built four-motor belt (firmware/belt/belt.ino). */
export const PINS4 = ["D26", "D14", "D13", "D4"];

/** Eight cells onto n motors.
 *  - 4: the firmware's constant-power fold. Each motor plays its own cell and 1/sqrt(2) of the
 *    two beside it, so a cue straight ahead runs both front motors.
 *  - 8 and 16: each motor plays the cells near its own bearing, weighted by a cosine that
 *    reaches zero 1.5 motor spacings away, so a cue between two motors runs both. */
export function spread(cells: Cell[], n: Layout): Cell[] {
  if (n === 4) {
    return [0, 1, 2, 3].map((m) => {
      const c = 2 * m + 1;
      const own = cells[c];
      const lt = cells[c - 1];
      const rt = cells[(c + 1) % 8];
      const level = Math.min(1, Math.sqrt(own.level ** 2 + 0.5 * (lt.level ** 2 + rt.level ** 2)));
      const parts: [Cell, number][] = [
        [own, own.level],
        [lt, lt.level * Math.SQRT1_2],
        [rt, rt.level * Math.SQRT1_2],
      ];
      const top = parts.reduce((a, b) => (b[1] > a[1] ? b : a));
      return { rgb: level > 0 ? top[0].rgb : null, level };
    });
  }
  const reach = n === 8 ? 60 : 30;
  return angles(n).map((a) => {
    let power = 0;
    let top: Cell | null = null;
    let best = 0;
    cells.forEach((c, i) => {
      const d = Math.abs(((a - i * 45 + 540) % 360) - 180);
      if (d >= reach) return;
      const v = c.level * Math.cos((d / reach) * (Math.PI / 2));
      power += v * v;
      if (v > best) [best, top] = [v, c];
    });
    const level = Math.min(1, Math.sqrt(power));
    return { rgb: level > 0 && top ? (top as Cell).rgb : null, level };
  });
}

/** How hard a motor buzzes as a colour: pale yellow when it barely runs, red at full power. */
export function heat(level: number): [number, number, number] {
  const k = Math.max(0, Math.min(1, level));
  const stops: [number, [number, number, number]][] = [
    [0, [255, 226, 110]],
    [0.5, [255, 150, 40]],
    [1, [255, 48, 36]],
  ];
  const j = k < 0.5 ? 0 : 1;
  const [k0, c0] = stops[j];
  const [k1, c1] = stops[j + 1];
  const f = (k - k0) / (k1 - k0);
  return [0, 1, 2].map((i) => Math.round(c0[i] + (c1[i] - c0[i]) * f)) as [number, number, number];
}
