import type { ReactNode } from "react";

export function Toggle({
  checked,
  onChange,
  disabled = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors ${
        checked ? "bg-[var(--color-real)]" : "bg-[var(--color-border)]"
      } ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
    >
      <span
        className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
          checked ? "translate-x-5" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}

/** One row in a settings panel: title + optional badge, description, and a
 * Toggle with an ON/OFF label -- the visual pattern requested for the
 * Analyze page's detection options. `highlighted` gives the row a tinted
 * background (used for the top-level option in a group, matching the
 * reference this was modeled on); `indented` nests a sub-option under it. */
export function SettingRow({
  title,
  description,
  badge,
  checked,
  onChange,
  disabled = false,
  highlighted = false,
  indented = false,
}: {
  title: string;
  description: ReactNode;
  badge?: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  highlighted?: boolean;
  indented?: boolean;
}) {
  return (
    <div
      className={`flex items-start justify-between gap-4 rounded-xl p-4 ${
        highlighted ? "bg-[var(--color-real-bg)]" : ""
      } ${indented ? "ml-4" : ""} ${disabled ? "opacity-60" : ""}`}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-[var(--color-text)]">{title}</span>
          {badge && (
            <span className="rounded bg-[var(--color-fake-bg)] px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-[var(--color-accent)] uppercase">
              {badge}
            </span>
          )}
        </div>
        <p className="mt-1 max-w-md text-xs text-[var(--color-muted)]">{description}</p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="text-xs font-semibold text-[var(--color-muted)]">{checked ? "ON" : "OFF"}</span>
        <Toggle checked={checked} onChange={onChange} disabled={disabled} />
      </div>
    </div>
  );
}
