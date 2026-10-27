import type { Safety } from "./types";

/** Stereo ticks that speed up as obstacles close in: the parts-free stand-in for the belt. */
export class AudioCues {
  private ctx: AudioContext | null = null;
  private on = false;
  private latest: Safety | null = null;
  private next = new Array(8).fill(0);
  private timer = 0;

  // Pan and pitch per sector: front sectors left to right, side sectors lower, rear lowest.
  private static PAN = [0, 0.5, 0.9, 0.5, 0, -0.5, -0.9, -0.5];
  private static PITCH = [880, 784, 660, 440, 330, 440, 660, 784];

  async toggle(): Promise<boolean> {
    this.on = !this.on;
    if (this.on) {
      this.ctx ??= new AudioContext();
      await this.ctx.resume();
      this.timer = window.setInterval(() => this.step(), 30);
    } else {
      window.clearInterval(this.timer);
    }
    return this.on;
  }

  update(s: Safety) {
    this.latest = s;
  }

  private tick(i: number, intensity: number) {
    const ctx = this.ctx!;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const pan = ctx.createStereoPanner();
    osc.frequency.value = AudioCues.PITCH[i];
    pan.pan.value = AudioCues.PAN[i];
    const t = ctx.currentTime;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.18 * intensity, t + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
    osc.connect(gain).connect(pan).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.09);
  }

  private step() {
    if (!this.latest || !this.ctx) return;
    const now = this.ctx.currentTime;
    this.latest.cues.forEach((c, i) => {
      if (c.mode === "off") return;
      const period = c.mode === "solid" ? 0.09 : 1 / c.rate_hz;
      if (now >= this.next[i]) {
        this.tick(i, c.intensity);
        this.next[i] = now + period;
      }
    });
  }
}
