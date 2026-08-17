export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-[var(--color-border)]"
      style={{
        width: size,
        height: size,
        borderTopColor: "var(--color-accent)",
      }}
    />
  );
}
