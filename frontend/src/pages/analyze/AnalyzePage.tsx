import { useState, type ReactNode } from "react";
import { Card } from "../../components/ui/Card";
import {
  ImageIcon,
  VideoIcon,
  UploadIcon,
  ScanIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  FaceScanIcon,
  SparkleIcon,
  FingerprintIcon,
} from "../../components/ui/Icons";
import { ImageAnalyzer } from "./ImageAnalyzer";
import { VideoAnalyzer } from "./VideoAnalyzer";

type Tab = "image" | "video";

export function AnalyzePage() {
  const [tab, setTab] = useState<Tab>("image");

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
          <ScanIcon width={15} height={15} />
        </span>
        <span className="font-mono text-[11px] font-bold tracking-[0.14em] text-[var(--color-accent)] uppercase">
          Analysis Workspace
        </span>
      </div>
      <h1 className="mt-3 font-display text-2xl font-bold tracking-tight text-[var(--color-text)] sm:text-[26px]">
        Analyze Media
      </h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
        Upload a photo or video for offline deepfake analysis — separate from the real-time call
        protection, useful for checking media you've already received.
      </p>

      <div className="mt-6 inline-flex rounded-xl bg-[var(--color-surface-raised)] p-1">
        <SegmentButton active={tab === "image"} onClick={() => setTab("image")} icon={<ImageIcon width={16} height={16} />}>
          Image
        </SegmentButton>
        <SegmentButton active={tab === "video"} onClick={() => setTab("video")} icon={<VideoIcon width={16} height={16} />}>
          Video
        </SegmentButton>
      </div>

      <WorkflowSteps />

      <div className="mt-5">{tab === "image" ? <ImageAnalyzer /> : <VideoAnalyzer />}</div>

      <WhatWeCheck />
    </div>
  );
}

function SegmentButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all duration-150 ${
        active
          ? "bg-[var(--color-surface)] text-[var(--color-text)] shadow-sm ring-1 ring-[var(--color-border)]"
          : "text-[var(--color-muted)] hover:bg-[var(--color-surface)]/60 hover:text-[var(--color-text)]"
      }`}
    >
      <span className={active ? "text-[var(--color-accent)]" : ""}>{icon}</span>
      {children}
    </button>
  );
}

const STEPS = [
  { icon: UploadIcon, label: "Upload" },
  { icon: ScanIcon, label: "Analyze" },
  { icon: CheckCircleIcon, label: "Results" },
];

function WorkflowSteps() {
  return (
    <div className="mt-5 flex flex-wrap items-center gap-1.5 text-xs font-medium text-[var(--color-muted)]">
      {STEPS.map((step, i) => (
        <div key={step.label} className="flex items-center gap-1.5">
          <div className="flex items-center gap-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5">
            <step.icon width={13} height={13} className="text-[var(--color-accent)]" />
            <span>{step.label}</span>
          </div>
          {i < STEPS.length - 1 && (
            <ArrowRightIcon width={13} height={13} className="text-[var(--color-border)]" />
          )}
        </div>
      ))}
    </div>
  );
}

const CHECKS = [
  { icon: FaceScanIcon, label: "Face authenticity" },
  { icon: ScanIcon, label: "Visual manipulation" },
  { icon: SparkleIcon, label: "Synthetic media indicators" },
  { icon: FingerprintIcon, label: "Identity consistency" },
];

function WhatWeCheck() {
  return (
    <Card className="mt-6 p-5">
      <div className="text-xs font-bold tracking-wide text-[var(--color-muted)] uppercase">
        What Trinetra Checks
      </div>
      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {CHECKS.map((item) => (
          <div key={item.label} className="flex items-center gap-2.5 text-sm text-[var(--color-text)]">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--color-accent)]/10 text-[var(--color-accent)]">
              <item.icon width={15} height={15} />
            </span>
            {item.label}
          </div>
        ))}
      </div>
    </Card>
  );
}
