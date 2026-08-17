import { NavLink, Outlet, Link } from "react-router-dom";
import { ScanIcon, HistoryIcon, UsersIcon, HomeUsersIcon, ShieldIcon } from "../ui/Icons";
import { useBackendStatus } from "../../lib/useBackendStatus";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Analyze", icon: ScanIcon, end: true },
  { to: "/dashboard/history", label: "History", icon: HistoryIcon, end: false },
  { to: "/dashboard/contacts", label: "Contacts", icon: UsersIcon, end: false },
  { to: "/dashboard/family", label: "Family Circles", icon: HomeUsersIcon, end: false },
];

export function DashboardLayout() {
  const status = useBackendStatus();

  const statusColor =
    status === "online"
      ? "var(--color-real)"
      : status === "offline"
        ? "var(--color-fake)"
        : "var(--color-muted)";
  const statusLabel =
    status === "online" ? "Backend online" : status === "offline" ? "Backend offline" : "Checking…";

  return (
    <div className="flex min-h-screen bg-[var(--color-bg)]">
      <aside className="flex w-60 shrink-0 flex-col border-r border-[var(--color-border)] px-4 py-6">
        <Link to="/" className="flex items-center gap-2 px-2 pb-8">
          <ShieldIcon className="text-[var(--color-accent)]" />
          <span className="font-display text-sm font-bold tracking-tight">Trinetra</span>
        </Link>

        <nav className="flex flex-1 flex-col gap-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-[var(--color-surface-raised)] text-[var(--color-text)]"
                    : "text-[var(--color-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]"
                }`
              }
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-[var(--color-muted)]">
          <span
            className="inline-block h-2 w-2 shrink-0 rounded-full"
            style={{ background: statusColor }}
          />
          {statusLabel}
        </div>
      </aside>

      <main className="min-w-0 flex-1 px-8 py-8">
        <div className="mx-auto max-w-4xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
