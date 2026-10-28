export interface Cue {
  mode: "off" | "slow" | "fast" | "solid";
  intensity: number;
  rate_hz: number;
}
export interface Obstacle {
  label: string;
  score: number;
  box: [number, number, number, number];
  dist: number | null;
}
export interface SegObject {
  label: string;
  score: number;
  poly: number[]; // normalised x, y pairs
  box: [number, number, number, number];
  dist: number | null;
}
export interface Safety {
  type: "safety";
  seq: number;
  ts: number;
  sectors: (number | null)[];
  names: string[];
  live: (number | null)[];
  cues: Cue[];
  obstacles: Obstacle[];
  objects?: SegObject[];
  prox?: string; // base64 uint8 grid, centimetres / 2, 255 = far or floor
  prox_size?: [number, number];
  fps: number;
  ms: Record<string, number>;
  scale: number;
  map_age: number | null;
}
export interface PinJson {
  id: number;
  label: string;
  kind: "object" | "landmark";
  pos: [number, number, number];
  ts: number;
  first_ts: number;
  seen: number;
  score: number;
  near: string[];
  thumb: string;
  source: string;
}
export interface Answer {
  type: "answer";
  label: string | null;
  text: string;
  pin: PinJson | null;
  language?: string;
}
export interface Alert {
  type: "alert";
  kind: "fall_prompt" | "fall_cancelled" | "fall_escalate" | "shake_notice";
  message: string;
}
export interface Pose {
  type: "pose";
  pos: [number, number, number];
  fwd: [number, number, number];
  up: [number, number, number];
}
export interface MemoryStats {
  type: "memory";
  window: number;
  views: number;
  infer_ms: number;
  total_ms: number;
  map_points: number;
  stitch_scale: number;
}
export interface Status {
  type: "status";
  memory?: string;
  finder?: string;
  memory_error?: string;
}
export type Message = Safety | Answer | Alert | Pose | MemoryStats | Status | PinsMsg | TrailMsg | ModeMsg;
export type Mode = "home" | "outside";
export interface ModeMsg {
  type: "mode";
  mode: Mode;
}
export interface PinsMsg {
  type: "pins";
  pins: PinJson[];
}
export interface TrailMsg {
  type: "trail";
  points: number[][];
}

/** Sector order matches the server: clockwise from straight ahead. */
export const SECTOR_NAMES = [
  "front",
  "front-right",
  "right",
  "back-right",
  "back",
  "back-left",
  "left",
  "front-left",
];
