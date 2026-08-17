export function StatTile({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="rounded-xl bg-[var(--color-surface-raised)] px-4 py-3">
      <div className="font-mono text-lg font-semibold text-[var(--color-text)]">{value}</div>
      <div className="text-xs text-[var(--color-muted)]">{label}</div>
    </div>
  );
}
