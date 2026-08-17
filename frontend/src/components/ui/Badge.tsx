import type { Classification } from "../../lib/types";
import { CLASSIFICATION_COLOR, CLASSIFICATION_LABEL } from "../../lib/classification";

const ICON: Record<Classification, string> = {
  real: "✓",
  fake: "⚠",
  uncertain: "?",
  insufficient_quality: "—",
  no_face_detected: "—",
};

export function ClassificationBadge({ classification }: { classification: Classification }) {
  const color = CLASSIFICATION_COLOR[classification];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 font-mono text-xs font-bold tracking-wide"
      style={{ color: color.fg, background: color.bg }}
    >
      <span>{ICON[classification]}</span>
      {CLASSIFICATION_LABEL[classification].toUpperCase()}
    </span>
  );
}
