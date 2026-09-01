import { NavLink, Outlet, Link } from "react-router-dom";
import { ScanIcon, HistoryIcon, UsersIcon, HomeUsersIcon, ShieldIcon, ReportIcon } from "../ui/Icons";
import { useBackendStatus } from "../../lib/useBackendStatus";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Analyze", icon: ScanIcon, end: true },
  { to: "/dashboard/history", label: "History", icon: HistoryIcon, end: false },
  { to: "/dashboard/reports", label: "Reports", icon: ReportIcon, end: false },
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
  const statusBg =
    status === "online"
      ? "var(--color-real-bg)"
      : status === "offline"
        ? "var(--color-fake-bg)"
        : "var(--color-neutral-bg)";
  const statusLabel =
    status === "online" ? "Backend online" : status === "offline" ? "Backend offline" : "Checking…";

  return (
    <div className="flex min-h-screen bg-[var(--color-bg)]">
      <aside className="flex w-60 shrink-0 flex-col border-r border-[var(--color-border)] px-4 py-6">
        <Link to="/" className="flex items-center gap-2.5 px-2 pb-5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
            <ShieldIcon width={17} height={17} />
          </span>
          <span className="min-w-0">
            <span className="block font-display text-sm leading-tight font-bold tracking-tight text-[var(--color-text)]">
              Trinetra
            </span>
            <span className="block font-mono text-[10px] leading-tight tracking-wide text-[var(--color-muted)] uppercase">
              Deepfake Detection
            </span>
          </span>
        </Link>

        <div className="border-t border-[var(--color-border)]" />

        <div className="mt-5 mb-1.5 px-3 font-mono text-[10px] font-semibold tracking-[0.12em] text-[var(--color-muted)] uppercase">
          Navigation
        </div>
        <nav className="flex flex-1 flex-col gap-0.5">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `group flex items-center gap-3 rounded-lg border-l-2 px-2.5 py-2.5 text-sm font-medium transition-colors duration-150 ${
                  isActive
                    ? "border-[var(--color-accent)] bg-[var(--color-accent)]/[0.07] text-[var(--color-text)]"
                    : "border-transparent text-[var(--color-muted)] hover:bg-[var(--color-surface-raised)]/60 hover:text-[var(--color-text)]"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    className={`flex h-5 w-5 shrink-0 items-center justify-center ${
                      isActive ? "text-[var(--color-accent)]" : "text-[var(--color-muted)] group-hover:text-[var(--color-text)]"
                    }`}
                  >
                    <Icon width={17} height={17} />
                  </span>
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium"
          style={{ background: statusBg, color: statusColor }}
        >
          <span
            className={`inline-block h-2 w-2 shrink-0 rounded-full ${status === "online" ? "animate-pulse" : ""}`}
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
