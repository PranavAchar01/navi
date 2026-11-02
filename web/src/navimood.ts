// What Navi feels, from what the HUD knows. Pure functions, so they are unit tested and the
// character (navibot.ts) only animates what this module decides.
//
//   mood    the lasting state: how the face rests
//   emote   a one-shot reaction to a single event
//   gaze    where the eyes look: toward the thing it is guiding to, or up at you

export type Mood = "idle" | "listening" | "thinking" | "guiding" | "careful" | "happy" | "worried" | "sleepy" | "offline";

export type Emote = "nod" | "hop" | "wince" | "surprise" | "delight" | "shrug";

export interface NaviInputs {
  online: boolean;
  /** The mic is open or the glasses are hearing a question. */
  listening: boolean;
  /** A question went out and no answer has come back yet. */
  waiting: boolean;
  /** The fall check-in is on screen. */
  alert: boolean;
  /** Seconds to contact with the nearest obstacle, or null. */
  ttc: number | null;
  /** Free path ahead in metres, or null. */
  path: number | null;
  /** Bearing to the thing being guided to (deg, clockwise from ahead), or null. */
  bearing: number | null;
  /** Milliseconds since the thing was found, when it was. */
  sinceFoundMs: number | null;
  /** Milliseconds since anything happened. */
  quietMs: number;
}

/** Close enough to an obstacle that Navi looks ahead and narrows its eyes. */
export function close(i: Pick<NaviInputs, "ttc" | "path">): boolean {
  return (i.ttc !== null && i.ttc < 1.6) || (i.path !== null && i.path < 0.7);
}

export function moodOf(i: NaviInputs): Mood {
  if (!i.online) return "offline";
  if (i.alert) return "worried";
  if (close(i)) return "careful";
  if (i.listening) return "listening";
  if (i.waiting) return "thinking";
  if (i.sinceFoundMs !== null && i.sinceFoundMs < 4000) return "happy";
  if (i.bearing !== null) return "guiding";
  if (i.quietMs > 60_000) return "sleepy";
  return "idle";
}

/** Where the eyes look, -1..1 each way: the guide bearing across, up at the wearer when
 *  listening, ahead and down when careful. Behind the wearer looks over a shoulder. */
export function gazeOf(mood: Mood, bearing: number | null): { x: number; y: number } {
  if (mood === "listening") return { x: 0, y: -0.7 };
  if (mood === "careful") return { x: 0, y: 0.5 };
  if (mood === "thinking") return { x: 0.55, y: -0.6 };
  if (bearing === null || mood === "sleepy" || mood === "offline") return { x: 0, y: 0 };
  const b = (((bearing % 360) + 540) % 360) - 180;
  const x = Math.abs(b) <= 90 ? Math.sin((b * Math.PI) / 180) : Math.sign(b);
  return { x, y: Math.abs(b) > 90 ? 0.35 : -0.1 };
}

/** One-shot reactions to the HUD's events. */
export function emoteFor(event: "answer" | "answer-none" | "found" | "obstacle" | "pin" | "alert"): Emote {
  return ({ answer: "nod", "answer-none": "shrug", found: "delight", obstacle: "wince", pin: "surprise", alert: "hop" } as const)[event];
}

const MAX = 64;
const short = (s: string) => (s.length > MAX ? s.slice(0, MAX - 1).trimEnd() + "…" : s);

/** What Navi says in its bubble: short and warm, never the safety numbers. */
export function lineFor(
  event: "hello" | "listening" | "thinking" | "found" | "answer" | "answer-none" | "offline" | "alert",
  label?: string,
): string {
  switch (event) {
    case "hello":
      return "Hi, I'm Navi. Ask me where anything is.";
    case "listening":
      return "I'm listening.";
    case "thinking":
      return "Let me remember…";
    case "found":
      return label ? short(`There it is, your ${label}!`) : "Found it!";
    case "answer":
      return label ? short(`I saw your ${label}. Follow the buzz.`) : "Follow the buzz.";
    case "answer-none":
      return "Hmm, I haven't seen that yet.";
    case "offline":
      return "I lost the server. Hang on.";
    case "alert":
      return "Are you okay?";
  }
}
