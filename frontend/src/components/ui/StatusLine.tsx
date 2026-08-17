import { Spinner } from "./Spinner";

export function StatusLine({
  uploading,
  processing,
  progress = 0,
  error,
}: {
  uploading?: boolean;
  processing?: boolean;
  progress?: number;
  error?: string | null;
}) {
  return (
    <div className="mt-3 space-y-2">
      {uploading && (
        <>
          <div className="h-1.5 overflow-hidden rounded-full bg-[var(--color-surface-raised)]">
            <div
              className="h-full bg-[var(--color-accent)] transition-[width]"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
          <div className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
            <Spinner size={13} /> Uploading… {Math.round(progress * 100)}%
          </div>
        </>
      )}
      {processing && (
        <div className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
          <Spinner size={13} /> Analyzing…
        </div>
      )}
      {error && (
        <div className="text-xs text-[var(--color-fake)]">⚠ {error}</div>
      )}
    </div>
  );
}
