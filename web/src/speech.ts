/** Automated runs (Playwright sets navigator.webdriver) and ?mute stay silent. */
const muted = navigator.webdriver || new URLSearchParams(location.search).has("mute");

export function speak(text: string): void {
  if (muted || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.02;
  window.speechSynthesis.speak(u);
}

interface Recognition {
  lang: string;
  interimResults: boolean;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  start(): void;
  stop(): void;
}

/** One utterance from the microphone (iOS Safari and Chrome expose it as webkitSpeechRecognition).
 * `onEnd(failed)`: failed when nothing was heard or the browser refused (an http page that is
 * not localhost gets "not-allowed"), so the caller can offer the text box instead. */
export function listen(onText: (t: string) => void, onEnd: (failed: boolean) => void): (() => void) | null {
  const Ctor = (window as unknown as { webkitSpeechRecognition?: new () => Recognition })
    .webkitSpeechRecognition;
  if (!Ctor) return null;
  const r = new Ctor();
  r.lang = "en-US";
  r.interimResults = false;
  let heard = false;
  let ended = false;
  const end = (failed: boolean) => {
    if (!ended) onEnd(failed);
    ended = true;
  };
  r.onresult = (e) => {
    heard = true;
    onText(e.results[0][0].transcript);
  };
  r.onend = () => end(!heard);
  r.onerror = () => end(true);
  try {
    r.start();
  } catch {
    return null;
  }
  return () => r.stop();
}
