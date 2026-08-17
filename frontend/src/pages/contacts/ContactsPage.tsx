import { useEffect, useState } from "react";
import { Card } from "../../components/ui/Card";
import { Spinner } from "../../components/ui/Spinner";
import { enrollContact, listContacts } from "../../lib/api";
import { formatDate } from "../../lib/classification";
import { ApiError } from "../../lib/types";
import type { Contact } from "../../lib/types";

export function ContactsPage() {
  const [contacts, setContacts] = useState<Contact[] | null>(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  function refresh() {
    listContacts()
      .then(setContacts)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load contacts"));
  }

  useEffect(refresh, []);

  async function handleSubmit() {
    if (!name.trim() || !file) {
      setError("Enter a name and choose a reference photo.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await enrollContact(name.trim(), file);
      setNotice(`Enrolled "${result.name}" — id ${result.id}`);
      setName("");
      setFile(null);
      refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Enrollment failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <h1 className="font-display text-xl font-bold">Contacts</h1>
      <p className="mt-1 text-sm text-[var(--color-muted)]">
        Add a reference photo for identity verification during calls — the real-time extension
        compares a live caller's face against these.
      </p>

      <Card className="mt-5 p-5">
        <h2 className="text-sm font-semibold">Enroll a Contact</h2>
        <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex-1 text-xs text-[var(--color-muted)]">
            Name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Mom"
              className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
            />
          </label>
          <label className="flex-1 text-xs text-[var(--color-muted)]">
            Reference photo
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="mt-1 w-full text-sm text-[var(--color-text)] file:mr-3 file:rounded-md file:border-0 file:bg-[var(--color-surface-raised)] file:px-3 file:py-1.5 file:text-xs file:text-[var(--color-text)]"
            />
          </label>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
          >
            {submitting ? <Spinner size={14} /> : "Enroll"}
          </button>
        </div>
        {error && <div className="mt-2 text-xs text-[var(--color-fake)]">⚠ {error}</div>}
        {notice && <div className="mt-2 text-xs text-[var(--color-real)]">{notice}</div>}
      </Card>

      <div className="mt-5">
        {!contacts && <div className="flex items-center gap-2 text-sm text-[var(--color-muted)]"><Spinner /> Loading…</div>}
        {contacts && contacts.length === 0 && (
          <Card className="p-6 text-center text-sm text-[var(--color-muted)]">No contacts enrolled yet.</Card>
        )}
        {contacts && contacts.length > 0 && (
          <div className="flex flex-col gap-2">
            {contacts.map((c) => (
              <Card key={c.id} className="flex items-center justify-between px-4 py-3">
                <span className="text-sm font-medium">{c.name}</span>
                <span className="text-xs text-[var(--color-muted)]">
                  #{c.id} · enrolled {formatDate(c.created_at)}
                </span>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
