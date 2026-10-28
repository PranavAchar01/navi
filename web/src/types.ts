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
export interface Hazard {
  poly: number[]; // normalised x, y pairs, smooth outline
  dist: number;
  tag: [number, number];
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
  hazards?: Hazard[];
  known: boolean[]; // false: no trusted data for that sector (stale map), the belt stays quiet
  path: number | null; // nearest obstacle in the 0.9 m walking path, metres
  trip: number | null; // nearest low obstacle (under 40 cm) in the path
  drop: number | null; // nearest drop below the floor in the path
  ttc: number | null; // seconds to contact at the current closing speed
  floor: boolean;
  floor_scale: number | null;
  eye_h: number;
  guide: { label: string; dist: number | null; bearing: number | null } | null;
  found: boolean;
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
/** What the belt's motors are doing, reported by the ESP32 (or its emulator) at 30 Hz. */
export interface Belt {
  type: "belt";
  device: string;
  lost: boolean;
  drop: boolean;
  found: boolean;
  duty: number[]; // 0..255 per motor, front first, clockwise
  modes: string[];
  age_ms: number | null;
  latency_ms: number | null;
}
/** The glasses (or their emulator) and their Bluetooth link to the phone. */
export interface Device {
  type: "device";
  name: string;
  state: string;
  res: [number, number];
  fps: number;
  imu_hz: number;
  kbps: number;
  budget_kbps: number;
  latency_ms?: number;
}
/** Speech heard by the glasses' microphones, partial until `final`. */
export interface Transcript {
  type: "transcript";
  text: string;
  final: boolean;
}
export type Message =
  | Safety
  | Belt
  | Answer
  | Alert
  | Pose
  | MemoryStats
  | Status
  | PinsMsg
  | TrailMsg
  | ModeMsg
  | Device
  | Transcript;
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
