import type { Net } from "./net";

const TARGET_FPS = 30; // the server keeps only the newest frame, so more is fresher, not slower
const LONG_SIDE = 640;

/** Frames and device motion from a phone camera or a video file, sent to the server. */
export class Source {
  private seq = 0;
  /** When each sent frame was grabbed (performance.now), so results can be drawn on their own frame. */
  readonly sentAt = new Map<number, number>();
  private running = false;
  private canvas = document.createElement("canvas");
  private motion: number[][] = [];
  private motionTimer = 0;

  constructor(
    private video: HTMLVideoElement,
    private net: Net,
  ) {}

  async startCamera(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();
    await this.startMotion();
    this.loop();
  }

  async startFile(file: File): Promise<void> {
    this.video.srcObject = null;
    this.video.src = URL.createObjectURL(file);
    this.video.loop = true;
    await this.video.play();
    this.loop();
  }

  private async startMotion(): Promise<void> {
    const DM = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> };
    if (typeof DM.requestPermission === "function") {
      if ((await DM.requestPermission()) !== "granted") return;
    }
    window.addEventListener("devicemotion", (e) => {
      const a = e.accelerationIncludingGravity;
      if (a && a.x !== null && a.y !== null && a.z !== null) {
        const sample = [e.timeStamp / 1000, a.x, a.y, a.z];
        // Gyro in rad/s about the device's x (right), y (up), z (out of the screen): the server
        // carries the head's turn with it between map poses. Spec: beta about x, gamma about y,
        // alpha about z, in degrees per second.
        const r = e.rotationRate;
        if (r && r.alpha !== null && r.beta !== null && r.gamma !== null) {
          const k = Math.PI / 180;
          sample.push(r.beta * k, r.gamma * k, r.alpha * k);
        }
        this.motion.push(sample);
      }
    });
    this.motionTimer = window.setInterval(() => {
      if (this.motion.length) {
        this.net.send({ type: "motion", s: this.motion });
        this.motion = [];
      }
    }, 100);
  }

  private loop(): void {
    if (this.running) return;
    this.running = true;
    const ctx = this.canvas.getContext("2d")!;
    let last = 0;
    const tick = (now: number) => {
      if (now - last >= 1000 / TARGET_FPS - 2 && this.net.backlog < 600_000) {
        last = now;
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        if (vw && vh && !this.video.paused) {
          const s = LONG_SIDE / Math.max(vw, vh);
          this.canvas.width = Math.round(vw * s);
          this.canvas.height = Math.round(vh * s);
          ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
          const grabbed = now;
          this.canvas.toBlob(
            async (b) => {
              if (!b) return;
              const seq = this.seq++;
              this.sentAt.set(seq, grabbed);
              this.sentAt.delete(seq - 300);
              this.net.sendFrame(seq, await b.arrayBuffer());
            },
            "image/jpeg",
            0.7,
          );
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    window.clearInterval(this.motionTimer);
  }
}
