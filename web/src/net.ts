import type { Message } from "./types";

type Handler<T> = (m: T) => void;

export class Net {
  private ws: WebSocket | null = null;
  private handlers = new Map<string, Handler<never>[]>();
  private onChunk: ((n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) => void) | null = null;
  private onFrame: ((seq: number, jpeg: Blob) => void) | null = null;
  private onReplace: ((n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) => void) | null = null;
  private onState: ((up: boolean) => void) | null = null;

  on<K extends Message["type"]>(type: K, fn: Handler<Extract<Message, { type: K }>>): void {
    const list = this.handlers.get(type) ?? [];
    list.push(fn as Handler<never>);
    this.handlers.set(type, list);
  }
  onMapChunk(fn: (n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) => void): void {
    this.onChunk = fn;
  }
  /** Camera frames another client sent, forwarded by the server after analysis. */
  onCameraFrame(fn: (seq: number, jpeg: Blob) => void): void {
    this.onFrame = fn;
  }
  onMapReplace(fn: (n: number, pts: Float32Array, cols: Uint8Array, sizes?: Uint8Array) => void): void {
    this.onReplace = fn;
  }
  onConnection(fn: (up: boolean) => void): void {
    this.onState = fn;
  }

  connect(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.onState?.(true);
    ws.onclose = () => {
      this.onState?.(false);
      setTimeout(() => this.connect(), 1500);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data) as Message;
        for (const fn of this.handlers.get(msg.type) ?? []) (fn as Handler<Message>)(msg);
      } else {
        const buf = ev.data as ArrayBuffer;
        const tag = new Uint8Array(buf, 0, 1)[0];
        if (tag === 70 /* 'F' */) {
          this.onFrame?.(new DataView(buf).getUint32(1, true), new Blob([buf.slice(5)], { type: "image/jpeg" }));
          return;
        }
        if (tag !== 77 /* 'M' */ && tag !== 82 /* 'R' */) return;
        const n = new DataView(buf).getUint32(1, true);
        // Offsets are not 4-aligned, so copy into aligned views.
        const pts = new Float32Array(buf.slice(5, 5 + n * 12));
        const cols = new Uint8Array(buf.slice(5 + n * 12, 5 + n * 15));
        // A solved map also carries one disc-size byte per point.
        const sizes = buf.byteLength === 5 + n * 16 ? new Uint8Array(buf.slice(5 + n * 15)) : undefined;
        (tag === 82 ? this.onReplace : this.onChunk)?.(n, pts, cols, sizes);
      }
    };
    this.ws = ws;
  }

  get open(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
  get backlog(): number {
    return this.ws?.bufferedAmount ?? 0;
  }
  send(obj: object): void {
    if (this.open) this.ws!.send(JSON.stringify(obj));
  }
  sendFrame(seq: number, jpeg: ArrayBuffer): void {
    if (!this.open) return;
    const out = new ArrayBuffer(12 + jpeg.byteLength);
    const dv = new DataView(out);
    dv.setFloat64(0, performance.timeOrigin + performance.now(), true);
    dv.setUint32(8, seq, true);
    new Uint8Array(out, 12).set(new Uint8Array(jpeg));
    this.ws!.send(out);
  }
}
