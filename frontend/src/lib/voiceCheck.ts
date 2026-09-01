// Client-side audio extraction + voice anti-spoofing check for the Video
// Analyzer's "Voice Check" option. Talks to voice-integrity/src/server.py
// (a separate service from the main backend -- see that file's docstring
// for why), which only accepts audio soundfile can read, not video
// containers -- so the audio track is decoded and re-encoded to WAV
// entirely in the browser via Web Audio API before upload, no server-side
// video handling needed for this.

export const VOICE_API_BASE = import.meta.env.VITE_VOICE_API_BASE_URL ?? "http://127.0.0.1:8001";

export interface VoiceAudioSegment {
  start: number;
  end: number;
  gustking: Record<string, number>;
  xlsr_sls: { bonafide: number; spoof: number };
  flagged: boolean;
}

/** Pre-enhancement scores, kept for comparison whenever `enhance` was
 * requested -- see server.py's /voice/check docstring and audio_enhance.py's
 * module docstring for why this is a second opinion, not a strict
 * improvement, and so is never silently dropped. */
export interface VoiceEnhancementInfo {
  applied: true;
  denoiser: string;
  enhancer: string;
  raw: {
    gustking: Record<string, number>;
    xlsr_sls: { bonafide: number; spoof: number };
    verdict: "FAKE/SPOOFED" | "LIKELY GENUINE";
  };
}

/** Speaker-identity match against a prior enrollVoice() upload -- answers
 * "is this the enrolled person's voice", independent of whether it's
 * synthetic. A convincing clone of the reference voice legitimately shows
 * same_speaker=true; pair with `verdict` for the spoof check. */
export interface VoiceIdentityMatch {
  reference_name: string;
  similarity: number;
  same_speaker: boolean;
}

export interface VoiceCheckResult {
  gustking: Record<string, number>;
  xlsr_sls: { bonafide: number; spoof: number };
  verdict: "FAKE/SPOOFED" | "LIKELY GENUINE";
  elapsed_s: number;
  /** Whole clip re-scored over overlapping ~4s windows, so a spoofed
   * segment inside an otherwise-genuine clip isn't averaged away. */
  segments: VoiceAudioSegment[];
  most_suspicious_segment: VoiceAudioSegment | null;
  /** Non-null only when checkVoice() was called with a referenceName. */
  identity: VoiceIdentityMatch | null;
  /** Non-null (and the fields above reflect the *enhanced* clip) only when
   * checkVoice() was called with enhance=true. */
  enhancement: VoiceEnhancementInfo | null;
}

function mixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const mono = new Float32Array(buffer.length);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) mono[i] += data[i] / buffer.numberOfChannels;
  }
  return mono;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** Decodes the audio track out of a video (or audio) file entirely
 * client-side and re-encodes it as a mono PCM WAV blob. Returns null if the
 * file has no decodable audio track (decodeAudioData throws for
 * video-only files, or codecs the browser's media pipeline can't decode). */
export async function extractAudioAsWav(file: File): Promise<Blob | null> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const audioContext = new AudioContextCtor();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const mono = mixToMono(audioBuffer);
    return encodeWav(mono, audioBuffer.sampleRate);
  } catch {
    return null;
  } finally {
    await audioContext.close();
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  let detail = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    detail = body.detail || detail;
  } catch {
    // not JSON -- keep the generic status message
  }
  return detail;
}

export async function checkVoice(
  wavBlob: Blob,
  enhance = false,
  referenceName?: string | null,
): Promise<VoiceCheckResult> {
  const formData = new FormData();
  formData.append("file", wavBlob, "audio.wav");
  if (enhance) formData.append("enhance", "true");
  if (referenceName) formData.append("reference_name", referenceName);

  const res = await fetch(`${VOICE_API_BASE}/voice/check`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

/** Uploads a reference voice sample under `name` for later speaker-identity
 * matching via checkVoice(..., referenceName). Re-enrolling the same name
 * overwrites the previous sample. */
export async function enrollVoice(name: string, fileOrBlob: Blob, filename = "reference.wav"): Promise<{ name: string }> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", fileOrBlob, filename);

  const res = await fetch(`${VOICE_API_BASE}/voice/enroll`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function listEnrolledVoices(): Promise<string[]> {
  const res = await fetch(`${VOICE_API_BASE}/voice/enrolled`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  const body = await res.json();
  return body.names;
}

export async function deleteEnrolledVoice(name: string): Promise<void> {
  const res = await fetch(`${VOICE_API_BASE}/voice/enrolled/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
}

/** URL for an <audio> element to play back an enrolled reference sample,
 * so the enrolled voice can be verified by ear, not just trusted by name. */
export function enrolledVoiceAudioUrl(name: string): string {
  return `${VOICE_API_BASE}/voice/enrolled/${encodeURIComponent(name)}/audio`;
}
