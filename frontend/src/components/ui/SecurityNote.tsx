import { LockIcon } from "./Icons";

/** Small trust signal shown near the upload area -- subtle by design, not
 * a marketing claim. Accurate to how analyze-image/analyze-video actually
 * work (backend/main.py): no history_db logging, video uploads land in a
 * throwaway tempfile.TemporaryDirectory cleaned up unconditionally after
 * the request, and everything runs on 127.0.0.1 -- the file itself never
 * leaves the machine. */
export function SecurityNote() {
  return (
    <div className="mt-3 flex items-center justify-center gap-1.5 text-[11px] text-[var(--color-muted)]">
      <LockIcon width={13} height={13} className="shrink-0" />
      Processed locally on this machine — files aren't stored or sent to a third party.
    </div>
  );
}
