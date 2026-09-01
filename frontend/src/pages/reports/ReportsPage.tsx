import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "../../components/ui/Card";
import { ClassificationBadge } from "../../components/ui/Badge";
import { Spinner } from "../../components/ui/Spinner";
import { StatTile } from "../../components/ui/StatTile";
import { DownloadIcon, ReportIcon, SearchIcon, SortIcon } from "../../components/ui/Icons";
import {
  classifyFakeProbability,
  CLASSIFICATION_COLOR,
  formatDate,
  parseSqliteUtc,
} from "../../lib/classification";
import { getHistory } from "../../lib/api";
import { buildHistoryReport } from "../../lib/report";
import { ApiError } from "../../lib/types";
import type { Classification, HistoryEntry } from "../../lib/types";

type ClassFilter = "all" | "real" | "fake" | "uncertain";
type SortOrder = "newest" | "oldest";

const CLASS_FILTERS: { value: ClassFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "real", label: "Real" },
  { value: "fake", label: "Fake" },
  { value: "uncertain", label: "Uncertain" },
];

/** "Today" / "Yesterday" / a full date -- used to group the list below,
 * same idea as an email inbox's date dividers. */
function dateGroupLabel(date: Date): string {
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diffDays = Math.round((startOfDay(new Date()) - startOfDay(date)) / 86_400_000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return date.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
}

export function ReportsPage() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [classFilter, setClassFilter] = useState<ClassFilter>("all");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");

  useEffect(() => {
    getHistory(50)
      .then(setEntries)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load history"));
  }, []);

  function downloadReport(entry: HistoryEntry) {
    const doc = buildHistoryReport(entry);
    doc.save(`trinetra-report-${entry.id}.pdf`);
  }

  // Summary reflects every fetched report, independent of the filters
  // below -- a stable count, not one that shrinks as you search/filter.
  const summary = useMemo(() => {
    const counts: Record<Classification, number> = {
      real: 0,
      fake: 0,
      uncertain: 0,
      insufficient_quality: 0,
      no_face_detected: 0,
    };
    let recent = 0;
    const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
    for (const e of entries ?? []) {
      counts[classifyFakeProbability(e.fake_probability)]++;
      if (parseSqliteUtc(e.created_at).getTime() >= dayAgo) recent++;
    }
    return { total: entries?.length ?? 0, recent, fake: counts.fake, real: counts.real };
  }, [entries]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = (entries ?? []).filter((e) => {
      if (classFilter !== "all" && classifyFakeProbability(e.fake_probability) !== classFilter) return false;
      if (q) {
        const haystack = `${e.platform ?? ""} ${e.identity_contact_name ?? ""}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });
    return sortOrder === "newest" ? list : [...list].reverse();
  }, [entries, search, classFilter, sortOrder]);

  // Consecutive same-day entries collapse under one date header -- `filtered`
  // is already sorted, so same-day entries are contiguous.
  const grouped = useMemo(() => {
    const groups: { label: string; items: HistoryEntry[] }[] = [];
    for (const e of filtered) {
      const label = dateGroupLabel(parseSqliteUtc(e.created_at));
      const last = groups[groups.length - 1];
      if (last && last.label === label) last.items.push(e);
      else groups.push({ label, items: [e] });
    }
    return groups;
  }, [filtered]);

  const hasActiveFilters = search.trim() !== "" || classFilter !== "all";

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
          <ReportIcon width={15} height={15} />
        </span>
        <span className="font-mono text-[11px] font-bold tracking-[0.14em] text-[var(--color-accent)] uppercase">
          Report Center
        </span>
      </div>
      <h1 className="mt-3 font-display text-2xl font-bold tracking-tight text-[var(--color-text)] sm:text-[26px]">
        Reports
      </h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
        Download a PDF report for any past detection. For a fuller report — per-frame timeline,
        per-face breakdown — run a scan on the Analyze page and download it right after.
      </p>

      {entries && entries.length > 0 && (
        <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatTile value={summary.total} label="total reports" />
          <StatTile value={summary.recent} label="last 24h" />
          <StatTile value={summary.fake} label="fake detections" />
          <StatTile value={summary.real} label="real detections" />
        </div>
      )}

      {entries && entries.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1">
            <SearchIcon
              width={14}
              height={14}
              className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-[var(--color-muted)]"
            />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search reports…"
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] py-2 pr-3 pl-8 text-xs text-[var(--color-text)] placeholder:text-[var(--color-muted)] focus:border-[var(--color-accent)]/50 focus:outline-none"
            />
          </div>

          <div className="inline-flex rounded-lg bg-[var(--color-surface-raised)] p-0.5">
            {CLASS_FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setClassFilter(f.value)}
                className={`rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors duration-150 ${
                  classFilter === f.value
                    ? "bg-[var(--color-surface)] text-[var(--color-text)] shadow-sm ring-1 ring-[var(--color-border)]"
                    : "text-[var(--color-muted)] hover:text-[var(--color-text)]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => setSortOrder((o) => (o === "newest" ? "oldest" : "newest"))}
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-2 text-xs font-medium text-[var(--color-text)] transition-colors hover:bg-[var(--color-surface-raised)]"
          >
            <SortIcon width={13} height={13} />
            {sortOrder === "newest" ? "Newest" : "Oldest"}
          </button>
        </div>
      )}

      <div className="mt-4">
        {error && (
          <Card className="border-[var(--color-fake)]/25 bg-[var(--color-fake-bg)] p-4 text-sm font-medium text-[var(--color-fake)]">
            ⚠ {error}
          </Card>
        )}

        {!entries && !error && (
          <Card className="flex items-center justify-center gap-2 p-10 text-sm text-[var(--color-muted)]">
            <Spinner /> Loading…
          </Card>
        )}

        {entries && entries.length === 0 && (
          <Card className="flex flex-col items-center gap-3 p-10 text-center">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--color-surface-raised)] text-[var(--color-muted)]">
              <ReportIcon width={20} height={20} />
            </span>
            <div className="max-w-xs text-sm text-[var(--color-muted)]">
              No reports yet. Reports generated from your detections will appear here.
            </div>
            <Link
              to="/dashboard"
              className="text-xs font-semibold text-[var(--color-accent)] hover:underline"
            >
              Go to Analyze
            </Link>
          </Card>
        )}

        {entries && entries.length > 0 && filtered.length === 0 && (
          <Card className="flex flex-col items-center gap-3 p-10 text-center">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--color-surface-raised)] text-[var(--color-muted)]">
              <SearchIcon width={20} height={20} />
            </span>
            <div className="max-w-xs text-sm text-[var(--color-muted)]">
              No reports match your filters.
            </div>
            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSearch("");
                  setClassFilter("all");
                }}
                className="text-xs font-semibold text-[var(--color-accent)] hover:underline"
              >
                Clear filters
              </button>
            )}
          </Card>
        )}

        {grouped.map((group) => (
          <div key={group.label} className="mb-5 last:mb-0">
            <div className="mb-2 px-1 font-mono text-[10px] font-bold tracking-[0.12em] text-[var(--color-muted)] uppercase">
              {group.label}
            </div>
            <div className="flex flex-col gap-2">
              {group.items.map((entry) => (
                <ReportRow key={entry.id} entry={entry} onDownload={downloadReport} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportRow({
  entry,
  onDownload,
}: {
  entry: HistoryEntry;
  onDownload: (entry: HistoryEntry) => void;
}) {
  const classification = classifyFakeProbability(entry.fake_probability);
  const color = CLASSIFICATION_COLOR[classification];

  return (
    <Card
      className="flex flex-wrap items-center gap-4 border-l-4 p-3 transition-colors duration-150 hover:bg-[var(--color-surface-raised)]/40"
      style={{ borderLeftColor: color.fg }}
    >
      <img
        src={`data:image/jpeg;base64,${entry.thumbnail_b64}`}
        alt=""
        className="h-14 w-14 shrink-0 rounded-lg object-cover"
      />

      <div className="min-w-0 flex-1">
        <ClassificationBadge classification={classification} />
        <div className="mt-1.5 truncate text-xs text-[var(--color-muted)]">
          {formatDate(entry.created_at)}
          {entry.platform ? ` · ${entry.platform}` : ""}
          {entry.identity_contact_name
            ? ` · matched ${entry.identity_contact_name} (${Math.round(
                (entry.identity_similarity ?? 0) * 100,
              )}%)`
            : ""}
        </div>
      </div>

      <div className="shrink-0 text-right">
        <div className="font-mono text-lg leading-tight font-bold" style={{ color: color.fg }}>
          {Math.round(entry.fake_probability * 100)}%
        </div>
        <div className="text-[10px] tracking-wide text-[var(--color-muted)] uppercase">fake score</div>
      </div>

      <button
        onClick={() => onDownload(entry)}
        title="Download PDF report"
        className="flex shrink-0 items-center gap-1.5 rounded-lg bg-[var(--color-accent)] px-3.5 py-2 text-xs font-semibold text-white transition-colors hover:bg-[var(--color-accent-hover)]"
      >
        <DownloadIcon width={14} height={14} />
        <span className="hidden sm:inline">Download PDF</span>
        <span className="sm:hidden">PDF</span>
      </button>
    </Card>
  );
}
