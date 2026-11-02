// Spring physics for Navi, from the Bara bot (bara/web/src/bot/spring.ts): every visible value is
// a damped spring toward a target, so a change of mood is a morph, never a cut.
export interface SpringCfg {
  stiffness: number;
  damping: number;
}

/** Stiffness / damping pairs (mass 1). Bouncy and wobbly overshoot on purpose. */
export const PRESETS = {
  smooth: { stiffness: 170, damping: 22 },
  gentle: { stiffness: 80, damping: 15 },
  snappy: { stiffness: 520, damping: 40 },
  bouncy: { stiffness: 320, damping: 13 },
  wobbly: { stiffness: 420, damping: 9 },
  blink: { stiffness: 1500, damping: 62 },
} satisfies Record<string, SpringCfg>;

export class Spring {
  value: number;
  target: number;
  velocity = 0;
  cfg: SpringCfg;

  constructor(value: number, cfg: SpringCfg = PRESETS.smooth) {
    this.value = value;
    this.target = value;
    this.cfg = cfg;
  }

  /** Kick the value without moving its target: squash, hop, nod. */
  impulse(v: number): void {
    this.velocity += v;
  }

  step(dt: number): void {
    const h = 1 / 240;
    for (let t = 0; t < dt; t += h) {
      const s = Math.min(h, dt - t);
      const a = -this.cfg.stiffness * (this.value - this.target) - this.cfg.damping * this.velocity;
      this.velocity += a * s;
      this.value += this.velocity * s;
    }
  }

  get settled(): boolean {
    return Math.abs(this.value - this.target) < 1e-3 && Math.abs(this.velocity) < 1e-3;
  }
}

export function reducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}


export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
