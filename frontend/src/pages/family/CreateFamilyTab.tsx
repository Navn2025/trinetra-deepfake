import { useState } from "react";
import { Card } from "../../components/ui/Card";
import { Spinner } from "../../components/ui/Spinner";
import { createFamily } from "../../lib/api";
import { ApiError } from "../../lib/types";

export function CreateFamilyTab() {
  const [name, setName] = useState("");
  const [passcode, setPasscode] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleCreate() {
    if (!name.trim() || !passcode) {
      setError("Enter a family name and passcode.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await createFamily(name.trim(), passcode);
      setNotice(
        `Created "${result.name}" — Family ID: ${result.id}. Share this ID + your passcode with family members so they can join.`,
      );
      setName("");
      setPasscode("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create family");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className="max-w-md p-5">
      <h2 className="text-sm font-semibold">Create a Family Circle</h2>
      <label className="mt-3 block text-xs text-[var(--color-muted)]">
        Family name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. The Sharmas"
          className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
        />
      </label>
      <label className="mt-3 block text-xs text-[var(--color-muted)]">
        Passcode (share this with family members)
        <input
          type="password"
          value={passcode}
          onChange={(e) => setPasscode(e.target.value)}
          placeholder="a shared secret"
          className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
        />
      </label>
      <button
        onClick={handleCreate}
        disabled={submitting}
        className="mt-4 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
      >
        {submitting ? <Spinner size={14} /> : "Create"}
      </button>
      {error && <div className="mt-3 text-xs text-[var(--color-fake)]">⚠ {error}</div>}
      {notice && <div className="mt-3 text-xs text-[var(--color-real)]">{notice}</div>}
    </Card>
  );
}
