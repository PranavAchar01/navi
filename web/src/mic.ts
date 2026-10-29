import type { Net } from "./net";

/** Stream the microphone to the server as 16 kHz mono `audio` messages (64 ms each), where
 * Muse Voice Transcribe hears the question and finds where it ends. Returns a stop function,
 * or null when the browser has no microphone to give. */
export async function streamMic(net: Net): Promise<(() => void) | null> {
  let media: MediaStream;
  try {
    media = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    return null;
  }
  const ctx = new AudioContext({ sampleRate: 16000 });
  const src = ctx.createMediaStreamSource(media);
  // ScriptProcessor is old but runs everywhere, Safari included, with no worklet file to serve.
  const proc = ctx.createScriptProcessor(1024, 1, 1);
  proc.onaudioprocess = (e) => {
    const f = e.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 32767;
    const bytes = new Uint8Array(pcm.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    net.send({ type: "audio", sr: ctx.sampleRate, ch: 1, pcm: btoa(bin) });
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  return () => {
    proc.disconnect();
    src.disconnect();
    media.getTracks().forEach((t) => t.stop());
    void ctx.close();
  };
}
