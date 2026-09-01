import { useRef, useState, type DragEvent } from "react";
import { UploadIcon } from "./Icons";

// Friendly labels for the mime types this app actually accepts (see
// ImageAnalyzer.tsx/VideoAnalyzer.tsx) -- derived by hand rather than
// parsed from the `accept` string so casing/naming stays consistent with
// the wording already used elsewhere (e.g. "MOV" not "QUICKTIME").
const FORMAT_LABELS: Record<string, string> = {
  "image/jpeg": "JPEG",
  "image/png": "PNG",
  "image/webp": "WebP",
  "video/mp4": "MP4",
  "video/webm": "WebM",
  "video/quicktime": "MOV",
  "video/x-matroska": "MKV",
};

export function Dropzone({
  accept,
  title,
  hint,
  onFile,
  maxSizeLabel,
}: {
  accept: string;
  title: string;
  hint: string;
  onFile: (file: File) => void;
  maxSizeLabel?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const formats = accept
    .split(",")
    .map((m) => FORMAT_LABELS[m.trim()])
    .filter((label): label is string => Boolean(label));

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragEnter={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
      className={`group relative cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all duration-200 ${
        dragOver
          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/[0.06] shadow-[0_0_0_4px_var(--color-fake-bg)]"
          : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/45 hover:bg-[var(--color-surface-raised)]/40"
      }`}
    >
      <div
        className={`mx-auto flex h-14 w-14 items-center justify-center rounded-2xl transition-all duration-200 ${
          dragOver
            ? "scale-110 bg-[var(--color-accent)] text-white"
            : "bg-[var(--color-accent)]/10 text-[var(--color-accent)] group-hover:scale-105 group-hover:bg-[var(--color-accent)]/15"
        }`}
      >
        <UploadIcon width={24} height={24} />
      </div>

      <div className="mt-4 text-[15px] font-semibold text-[var(--color-text)]">
        {dragOver ? "Drop to upload" : title}
      </div>
      <div className="mt-1 text-xs text-[var(--color-muted)]">{hint}</div>

      {(formats.length > 0 || maxSizeLabel) && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-1.5">
          {formats.map((f) => (
            <span
              key={f}
              className="rounded-md bg-[var(--color-surface-raised)] px-2 py-0.5 font-mono text-[10px] font-semibold tracking-wide text-[var(--color-muted)]"
            >
              {f}
            </span>
          ))}
          {maxSizeLabel && (
            <span className="rounded-md bg-[var(--color-surface-raised)] px-2 py-0.5 font-mono text-[10px] font-semibold tracking-wide text-[var(--color-muted)]">
              {maxSizeLabel}
            </span>
          )}
        </div>
      )}

      <button
        type="button"
        tabIndex={-1}
        className="mt-5 inline-flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-2 text-xs font-semibold text-[var(--color-text)] transition-colors group-hover:border-[var(--color-accent)]/40 group-hover:text-[var(--color-accent)]"
      >
        Browse Files
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
        }}
      />
    </div>
  );
}
