/** How the belt's eight protocol cells land on a belt of 4, 8 or 16 motors, for the 3D belt.
 *  Pure, so the tests run without WebGL. */

export type Layout = 4 | 8 | 16;
export const LAYOUTS: Layout[] = [4, 8, 16];

/** One motor's look: its colour (null = dark) and how hard it runs right now, 0..1. */
export interface Cell {
  rgb: [number, number, number] | null;
  level: number;
}

/** Motor angles, degrees clockwise from the front. Four is the built belt: front-right (D26),
 *  back-right (D14), back-left (D13), front-left (D4). */
export function angles(n: Layout): number[] {
  if (n === 4) return [45, 135, 225, 315];
  return Array.from({ length: n }, (_, i) => (i * 360) / n);
}

/** GPIO of each motor on the built four-motor belt (firmware/belt/belt.ino). */
export const PINS4 = ["D26", "D14", "D13", "D4"];

/** Eight cells onto n motors.
 *  - 4: the firmware's constant-power fold. Each motor plays its own cell and 1/sqrt(2) of the
 *    two beside it, so a cue straight ahead runs both front motors.
 *  - 8: one motor per cell.
 *  - 16: a motor on each cell, plus one between each pair that runs only when both neighbours
 *    do, so a cue between two cells lands between them. */
export function spread(cells: Cell[], n: Layout): Cell[] {
  if (n === 8) return cells;
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
  const out: Cell[] = [];
  for (let i = 0; i < 8; i++) {
    const a = cells[i];
    const b = cells[(i + 1) % 8];
    out.push(a);
    const level = Math.min(a.level, b.level);
    out.push({ rgb: level > 0 ? (a.level >= b.level ? a.rgb : b.rgb) : null, level });
  }
  return out;
}
