import { useState } from "react";
import { ImageIcon, VideoIcon } from "../../components/ui/Icons";
import { ImageAnalyzer } from "./ImageAnalyzer";
import { VideoAnalyzer } from "./VideoAnalyzer";

type Tab = "image" | "video";

export function AnalyzePage() {
  const [tab, setTab] = useState<Tab>("image");

  return (
    <div>
      <h1 className="font-display text-xl font-bold">Analyze Media</h1>
      <p className="mt-1 text-sm text-[var(--color-muted)]">
        Upload a photo or video for offline deepfake analysis — separate from the real-time call
        protection, useful for checking media you've already received.
      </p>

      <div className="mt-5 flex gap-2">
        <TabButton active={tab === "image"} onClick={() => setTab("image")} icon={<ImageIcon />}>
          Image
        </TabButton>
        <TabButton active={tab === "video"} onClick={() => setTab("video")} icon={<VideoIcon />}>
          Video
        </TabButton>
      </div>

      <div className="mt-4">
        {tab === "image" ? <ImageAnalyzer /> : <VideoAnalyzer />}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-semibold transition-colors ${
        active
          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]"
          : "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-muted)] hover:text-[var(--color-text)]"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}
