import type { ReactNode } from "react";

/** One cell in a connected statistics strip (see HistoryPage.tsx) -- richer
 * than StatTile (icon + semantic color), used where a handful of related
 * counts should read as one section rather than separate boxes. Cells are
 * meant to sit in a `grid gap-px bg-[var(--color-border)]` wrapper so the
 * 1px gaps show through as dividers between them. */
export function StatCard({
  icon,
  value,
  label,
  color = "var(--color-text)",
  background = "var(--color-surface-raised)",
}: {
  icon: ReactNode;
  value: string | number;
  label: string;
  color?: string;
  background?: string;
}) {
  return (
    <div className="flex items-center gap-3 bg-[var(--color-surface)] px-4 py-3.5">
      <span
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
        style={{ background, color }}
      >
        {icon}
      </span>
      <div className="min-w-0">
        <div className="font-mono text-lg leading-tight font-bold text-[var(--color-text)]">{value}</div>
        <div className="truncate text-[11px] text-[var(--color-muted)]">{label}</div>
      </div>
    </div>
  );
}
