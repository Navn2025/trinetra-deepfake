import { useState } from "react";
import { Card } from "../../components/ui/Card";
import { Spinner } from "../../components/ui/Spinner";
import { approveMember, getFamilyMembers, rejectMember } from "../../lib/api";
import { formatDate } from "../../lib/classification";
import { ApiError } from "../../lib/types";
import type { FamilyMember } from "../../lib/types";

const STATUS_COLOR: Record<FamilyMember["status"], string> = {
  pending: "var(--color-uncertain)",
  approved: "var(--color-real)",
  rejected: "var(--color-fake)",
};

export function ApproveMembersTab() {
  const [familyId, setFamilyId] = useState("");
  const [passcode, setPasscode] = useState("");
  const [members, setMembers] = useState<FamilyMember[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actingId, setActingId] = useState<number | null>(null);

  async function loadMembers() {
    if (!familyId || !passcode) {
      setError("Enter family ID and passcode.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setMembers(await getFamilyMembers(Number(familyId), passcode));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load members");
    } finally {
      setLoading(false);
    }
  }

  async function act(memberId: number, action: "approve" | "reject") {
    setActingId(memberId);
    try {
      const fn = action === "approve" ? approveMember : rejectMember;
      await fn(Number(familyId), memberId, passcode);
      setMembers(await getFamilyMembers(Number(familyId), passcode));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Action failed");
    } finally {
      setActingId(null);
    }
  }

  return (
    <Card className="max-w-xl p-5">
      <h2 className="text-sm font-semibold">Approve Members</h2>
      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
        <label className="flex-1 text-xs text-[var(--color-muted)]">
          Family ID
          <input
            type="number"
            value={familyId}
            onChange={(e) => setFamilyId(e.target.value)}
            className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
          />
        </label>
        <label className="flex-1 text-xs text-[var(--color-muted)]">
          Passcode
          <input
            type="password"
            value={passcode}
            onChange={(e) => setPasscode(e.target.value)}
            className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
          />
        </label>
        <button
          onClick={loadMembers}
          disabled={loading}
          className="rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
        >
          {loading ? <Spinner size={14} /> : "Load Members"}
        </button>
      </div>

      {error && <div className="mt-3 text-xs text-[var(--color-fake)]">⚠ {error}</div>}

      {members && (
        <div className="mt-4 flex flex-col gap-2">
          {members.length === 0 && (
            <div className="text-sm text-[var(--color-muted)]">No members yet.</div>
          )}
          {members.map((m) => (
            <div
              key={m.id}
              className="flex items-center justify-between rounded-lg bg-[var(--color-surface-raised)] px-3 py-2.5"
            >
              <div>
                <div className="flex items-center gap-2 text-sm font-medium">
                  {m.name}
                  <span
                    className="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase text-white"
                    style={{ background: STATUS_COLOR[m.status] }}
                  >
                    {m.status}
                  </span>
                </div>
                <div className="mt-0.5 text-xs text-[var(--color-muted)]">
                  requested {formatDate(m.created_at)}
                  {m.approved_at ? ` · approved ${formatDate(m.approved_at)}` : ""}
                  {!m.has_face ? " · no face captured" : ""}
                </div>
              </div>
              {m.status === "pending" && (
                <div className="flex gap-2">
                  <button
                    onClick={() => act(m.id, "approve")}
                    disabled={actingId === m.id}
                    className="rounded-md bg-[var(--color-real)]/15 px-2.5 py-1 text-xs font-semibold text-[var(--color-real)] hover:bg-[var(--color-real)]/25 disabled:opacity-50"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => act(m.id, "reject")}
                    disabled={actingId === m.id}
                    className="rounded-md bg-[var(--color-fake)]/15 px-2.5 py-1 text-xs font-semibold text-[var(--color-fake)] hover:bg-[var(--color-fake)]/25 disabled:opacity-50"
                  >
                    Reject
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
