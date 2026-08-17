import { useRef, useState, type DragEvent } from "react";

export function Dropzone({
  accept,
  title,
  hint,
  onFile,
}: {
  accept: string;
  title: string;
  hint: string;
  onFile: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

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
      className={`cursor-pointer rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
        dragOver
          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/5"
          : "border-[var(--color-border)] hover:border-[var(--color-muted)]"
      }`}
    >
      <div className="text-sm text-[var(--color-text)]">{title}</div>
      <div className="mt-1 text-xs text-[var(--color-muted)]">{hint}</div>
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
