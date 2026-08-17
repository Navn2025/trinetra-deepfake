import { useEffect, useState } from "react";
import { Card } from "../../components/ui/Card";
import { ClassificationBadge } from "../../components/ui/Badge";
import { Spinner } from "../../components/ui/Spinner";
import { classifyFakeProbability, fakeScorePct, formatDate } from "../../lib/classification";
import { getHistory } from "../../lib/api";
import { ApiError } from "../../lib/types";
import type { HistoryEntry } from "../../lib/types";

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory(50)
      .then(setEntries)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load history"));
  }, []);

  return (
    <div>
      <h1 className="font-display text-xl font-bold">History</h1>
      <p className="mt-1 text-sm text-[var(--color-muted)]">
        Recent real-time predictions from the browser extension / desktop app, newest first.
      </p>

      <div className="mt-5">
        {error && <div className="text-sm text-[var(--color-fake)]">⚠ {error}</div>}

        {!entries && !error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
            <Spinner /> Loading…
          </div>
        )}

        {entries && entries.length === 0 && (
          <Card className="p-6 text-center text-sm text-[var(--color-muted)]">
            No predictions yet. This fills up once the browser extension or desktop app runs a live
            check.
          </Card>
        )}

        {entries && entries.length > 0 && (
          <div className="flex flex-col gap-2">
            {entries.map((e) => {
              const classification = classifyFakeProbability(e.fake_probability);
              return (
                <Card key={e.id} className="flex items-center gap-4 p-3">
                  <img
                    src={`data:image/jpeg;base64,${e.thumbnail_b64}`}
                    alt=""
                    className="h-14 w-14 shrink-0 rounded-lg object-cover"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <ClassificationBadge classification={classification} />
                      <span className="text-xs text-[var(--color-muted)]">
                        {fakeScorePct(e.fake_probability)}
                      </span>
                    </div>
                    <div className="mt-1 truncate text-xs text-[var(--color-muted)]">
                      {formatDate(e.created_at)}
                      {e.platform ? ` · ${e.platform}` : ""}
                      {e.identity_contact_name
                        ? ` · matched ${e.identity_contact_name} (${Math.round(
                            (e.identity_similarity ?? 0) * 100,
                          )}%)`
                        : ""}
                    </div>
                  </div>
                  {e.scam_likelihood !== null && (
                    <div className="shrink-0 text-right">
                      <div className="text-sm font-semibold">
                        {Math.round(e.scam_likelihood * 100)}%
                      </div>
                      <div className="text-[10px] text-[var(--color-muted)]">scam likelihood</div>
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
