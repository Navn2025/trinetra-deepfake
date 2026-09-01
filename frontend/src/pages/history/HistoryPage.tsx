import { useEffect, useMemo, useState } from "react";
import { Card } from "../../components/ui/Card";
import { ClassificationBadge } from "../../components/ui/Badge";
import { Spinner } from "../../components/ui/Spinner";
import { StatCard } from "../../components/ui/StatCard";
import { ScoreRing } from "../../components/ui/ScoreRing";
import {
  HistoryIcon,
  ChecksIcon,
  CheckCircleIcon,
  AlertTriangleIcon,
  HelpCircleIcon,
  SearchIcon,
  SortIcon,
} from "../../components/ui/Icons";
import {
  classifyFakeProbability,
  CLASSIFICATION_COLOR,
  formatDate,
} from "../../lib/classification";
import { getHistory } from "../../lib/api";
import { ApiError } from "../../lib/types";
import type { Classification, HistoryEntry } from "../../lib/types";

type ClassFilter = "all" | "real" | "fake" | "uncertain";
type SourceFilter = "all" | "extension" | "desktop";
type SortOrder = "newest" | "oldest";

const CLASS_FILTERS: { value: ClassFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "real", label: "Real" },
  { value: "fake", label: "Fake" },
  { value: "uncertain", label: "Uncertain" },
];

/** The extension sends `window.location.hostname` as platform (e.g.
 * "web.whatsapp.com"); the desktop app sends `f"desktop:{title}"` (see
 * desktop/main.py) -- that prefix is the only real signal distinguishing
 * the two sources in the data we actually have. */
function sourceOf(platform: string | null): "extension" | "desktop" | "unknown" {
  if (!platform) return "unknown";
  return platform.startsWith("desktop:") ? "desktop" : "extension";
}

function sourceLabel(platform: string | null): string {
  if (!platform) return "Unknown source";
  return platform.startsWith("desktop:") ? platform.replace(/^desktop:/, "Desktop — ") : platform;
}

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [classFilter, setClassFilter] = useState<ClassFilter>("all");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");

  useEffect(() => {
    getHistory(50)
      .then(setEntries)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load history"));
  }, []);

  // Stats reflect the full fetched set regardless of the filters below --
  // a stable page-level summary, not a moving target as you narrow the list.
  const counts = useMemo(() => {
    const result: Record<Classification, number> = {
      real: 0,
      fake: 0,
      uncertain: 0,
      insufficient_quality: 0,
      no_face_detected: 0,
    };
    for (const e of entries ?? []) result[classifyFakeProbability(e.fake_probability)]++;
    return result;
  }, [entries]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = (entries ?? []).filter((e) => {
      if (classFilter !== "all" && classifyFakeProbability(e.fake_probability) !== classFilter) return false;
      if (sourceFilter !== "all" && sourceOf(e.platform) !== sourceFilter) return false;
      if (q) {
        const haystack = `${e.platform ?? ""} ${e.identity_contact_name ?? ""}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      return true;
    });
    return sortOrder === "newest" ? list : [...list].reverse();
  }, [entries, search, classFilter, sourceFilter, sortOrder]);

  const hasActiveFilters = search.trim() !== "" || classFilter !== "all" || sourceFilter !== "all";

  function clearFilters() {
    setSearch("");
    setClassFilter("all");
    setSourceFilter("all");
  }

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
          <HistoryIcon width={15} height={15} />
        </span>
        <span className="font-mono text-[11px] font-bold tracking-[0.14em] text-[var(--color-accent)] uppercase">
          Detection Log
        </span>
      </div>
      <h1 className="mt-3 font-display text-2xl font-bold tracking-tight text-[var(--color-text)] sm:text-[26px]">
        History
      </h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
        Recent real-time predictions from the browser extension and desktop app, newest first.
      </p>

      {entries && entries.length > 0 && (
        <Card className="mt-5 overflow-hidden p-0">
          <div className="grid grid-cols-2 gap-px bg-[var(--color-border)] sm:grid-cols-4">
            <StatCard icon={<ChecksIcon width={17} height={17} />} value={entries.length} label="Total checks" />
            <StatCard
              icon={<CheckCircleIcon width={17} height={17} />}
              value={counts.real}
              label="Real"
              color={CLASSIFICATION_COLOR.real.fg}
              background={CLASSIFICATION_COLOR.real.bg}
            />
            <StatCard
              icon={<AlertTriangleIcon width={17} height={17} />}
              value={counts.fake}
              label="Fake"
              color={CLASSIFICATION_COLOR.fake.fg}
              background={CLASSIFICATION_COLOR.fake.bg}
            />
            <StatCard
              icon={<HelpCircleIcon width={17} height={17} />}
              value={counts.uncertain}
              label="Uncertain"
              color={CLASSIFICATION_COLOR.uncertain.fg}
              background={CLASSIFICATION_COLOR.uncertain.bg}
            />
          </div>
        </Card>
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
              placeholder="Search detection history…"
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

          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value as SourceFilter)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-2 text-xs font-medium text-[var(--color-text)] focus:border-[var(--color-accent)]/50 focus:outline-none"
          >
            <option value="all">All sources</option>
            <option value="extension">Browser extension</option>
            <option value="desktop">Desktop app</option>
          </select>

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
              <HistoryIcon width={20} height={20} />
            </span>
            <div className="max-w-xs text-sm text-[var(--color-muted)]">
              No predictions yet. This fills up once the browser extension or desktop app runs a live
              check.
            </div>
          </Card>
        )}

        {entries && entries.length > 0 && filtered.length === 0 && (
          <Card className="flex flex-col items-center gap-3 p-10 text-center">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--color-surface-raised)] text-[var(--color-muted)]">
              <SearchIcon width={20} height={20} />
            </span>
            <div className="max-w-xs text-sm text-[var(--color-muted)]">
              No detections match your filters.
            </div>
            {hasActiveFilters && (
              <button
                onClick={clearFilters}
                className="text-xs font-semibold text-[var(--color-accent)] hover:underline"
              >
                Clear filters
              </button>
            )}
          </Card>
        )}

        {filtered.length > 0 && (
          <div className="flex flex-col gap-2">
            {filtered.map((e) => {
              const classification = classifyFakeProbability(e.fake_probability);
              const color = CLASSIFICATION_COLOR[classification];
              const highScam = e.scam_likelihood !== null && e.scam_likelihood >= 0.5;

              return (
                <Card
                  key={e.id}
                  className="flex flex-wrap items-center gap-4 p-3 transition-colors duration-150 hover:border-[var(--color-accent)]/25 hover:bg-[var(--color-surface-raised)]/40"
                >
                  <img
                    src={`data:image/jpeg;base64,${e.thumbnail_b64}`}
                    alt=""
                    className="h-14 w-14 shrink-0 rounded-lg object-cover ring-1 ring-[var(--color-border)]"
                  />

                  <div className="min-w-0 flex-1">
                    <ClassificationBadge classification={classification} />
                    <div className="mt-1.5 truncate text-xs text-[var(--color-muted)]">
                      {formatDate(e.created_at)} · {sourceLabel(e.platform)}
                      {e.identity_contact_name
                        ? ` · matched ${e.identity_contact_name} (${Math.round(
                            (e.identity_similarity ?? 0) * 100,
                          )}%)`
                        : ""}
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-3">
                    {e.scam_likelihood !== null && highScam && (
                      <span className="rounded-md bg-[var(--color-fake-bg)] px-2 py-1 text-[10px] font-semibold text-[var(--color-fake)]">
                        {Math.round(e.scam_likelihood * 100)}% scam risk
                      </span>
                    )}
                    <ScoreRing value={e.fake_probability} color={color.fg} />
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
