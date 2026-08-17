import type { Classification } from "./types";

// Mirrors config.py's FAKE_THRESHOLD/REAL_THRESHOLD defaults (0.65/0.35).
// The /history endpoint only returns a raw fake_probability, not a
// classification -- this reproduces the same bucketing client-side for
// display. If the backend is run with FAKE_THRESHOLD/REAL_THRESHOLD env
// overrides, this won't reflect that; it's a display convenience, not the
// source of truth (the backend's own classify_fake_probability is that).
export function classifyFakeProbability(p: number): Classification {
  if (p >= 0.65) return "fake";
  if (p <= 0.35) return "real";
  return "uncertain";
}

export const CLASSIFICATION_LABEL: Record<Classification, string> = {
  real: "Real",
  fake: "Fake",
  uncertain: "Uncertain",
  insufficient_quality: "Insufficient Quality",
  no_face_detected: "No Face Detected",
};

export const CLASSIFICATION_COLOR: Record<Classification, { fg: string; bg: string }> = {
  real: { fg: "var(--color-real)", bg: "var(--color-real-bg)" },
  fake: { fg: "var(--color-fake)", bg: "var(--color-fake-bg)" },
  uncertain: { fg: "var(--color-uncertain)", bg: "var(--color-uncertain-bg)" },
  insufficient_quality: { fg: "var(--color-neutral)", bg: "var(--color-neutral-bg)" },
  no_face_detected: { fg: "var(--color-neutral)", bg: "var(--color-neutral-bg)" },
};

/** Always the raw fake_probability, regardless of which classification this
 * is shown next to -- "Real 17%" reads as "17% confident real" when it
 * actually means "17% fake score, classified real because that's under
 * REAL_THRESHOLD". Every caller labels it explicitly as a fake score. */
export function fakeScorePct(p: number | null): string {
  return p === null ? "—" : `${Math.round(p * 100)}% fake score`;
}

export function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

/** SQLite's datetime('now') stores UTC as "YYYY-MM-DD HH:MM:SS" with no
 * timezone marker -- Date() would otherwise parse that as local time. */
export function formatDate(sqliteUtc: string): string {
  const iso = sqliteUtc.includes("T") ? sqliteUtc : sqliteUtc.replace(" ", "T");
  const date = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
