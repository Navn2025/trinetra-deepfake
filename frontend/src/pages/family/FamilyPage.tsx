import { useState } from "react";
import { CreateFamilyTab } from "./CreateFamilyTab";
import { JoinFamilyTab } from "./JoinFamilyTab";
import { ApproveMembersTab } from "./ApproveMembersTab";

type Tab = "create" | "join" | "approve";

export function FamilyPage() {
  const [tab, setTab] = useState<Tab>("create");

  const tabs: { id: Tab; label: string }[] = [
    { id: "create", label: "Create Family" },
    { id: "join", label: "Join Family" },
    { id: "approve", label: "Approve Members" },
  ];

  return (
    <div>
      <h1 className="font-display text-xl font-bold">Family Circle</h1>
      <p className="mt-1 text-sm text-[var(--color-muted)]">
        Enroll trusted family members' faces so calls can be checked against who they claim to be,
        not just whether the video looks AI-generated.
      </p>

      <div className="mt-5 flex gap-1 border-b border-[var(--color-border)]">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              tab === t.id
                ? "border-[var(--color-accent)] text-[var(--color-text)]"
                : "border-transparent text-[var(--color-muted)] hover:text-[var(--color-text)]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-5">
        {tab === "create" && <CreateFamilyTab />}
        {tab === "join" && <JoinFamilyTab />}
        {tab === "approve" && <ApproveMembersTab />}
      </div>
    </div>
  );
}
