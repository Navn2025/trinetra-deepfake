import { useRef, useState } from "react";
import { Dropzone } from "../../components/ui/Dropzone";
import { StatusLine } from "../../components/ui/StatusLine";
import { Card } from "../../components/ui/Card";
import { StatTile } from "../../components/ui/StatTile";
import { ClassificationBadge } from "../../components/ui/Badge";
import { fakeScorePct, formatTimestamp } from "../../lib/classification";
import { analyzeVideo } from "../../lib/api";
import { ApiError } from "../../lib/types";
import type { AnalyzeVideoResult } from "../../lib/types";

const STRIP_COLOR: Record<string, string> = {
  real: "var(--color-real)",
  fake: "var(--color-fake)",
  uncertain: "var(--color-uncertain)",
  insufficient_quality: "var(--color-neutral)",
  no_face_detected: "var(--color-neutral)",
};

export function VideoAnalyzer() {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeVideoResult | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  async function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    setResult(null);
    setError(null);
    setUploading(true);
    setProgress(0);
    setProcessing(false);

    try {
      const res = await analyzeVideo(file, (p) => {
        setProgress(p);
        if (p >= 1) {
          setUploading(false);
          setProcessing(true);
        }
      });
      setUploading(false);
      setProcessing(false);
      setResult(res);
    } catch (err) {
      setUploading(false);
      setProcessing(false);
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    }
  }

  function seekTo(t: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      videoRef.current.play();
    }
  }

  return (
    <div>
      <Dropzone
        accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
        title="Drop a video here, or click to choose a file"
        hint="MP4, WebM, MOV, or MKV — frames are sampled evenly across the clip"
        onFile={handleFile}
      />

      {objectUrl && (
        <Card className="mt-4 p-4">
          <video ref={videoRef} src={objectUrl} controls className="block w-full rounded-lg" />

          <StatusLine uploading={uploading} processing={processing} progress={progress} error={error} />

          {result && <ResultPanel result={result} onSeek={seekTo} />}
        </Card>
      )}
    </div>
  );
}

function ResultPanel({
  result,
  onSeek,
}: {
  result: AnalyzeVideoResult;
  onSeek: (t: number) => void;
}) {
  const m = result.metadata;
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ClassificationBadge classification={result.overall_classification} />
          <span className="text-xs text-[var(--color-muted)]">
            {fakeScorePct(result.overall_confidence)}
          </span>
        </div>
        <div className="text-xs text-[var(--color-muted)]">
          {result.processing_time_ms}ms{result.model_version ? ` · ${result.model_version}` : ""}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <StatTile value={`${m.duration_seconds}s`} label="duration" />
        <StatTile value={m.fps} label="native fps" />
        <StatTile value={`${m.width}×${m.height}`} label="resolution" />
        <StatTile value={m.frames_analyzed} label="frames sampled" />
        <StatTile value={result.suspicious_segments.length} label="suspicious segment(s)" />
      </div>

      <div className="mt-3 flex h-8 gap-0.5">
        {result.frames.map((f, i) => (
          <div
            key={i}
            role="button"
            title={`${formatTimestamp(f.timestamp)} — ${f.classification}${
              f.fake_probability !== null ? " " + fakeScorePct(f.fake_probability) : ""
            }`}
            onClick={() => onSeek(f.timestamp)}
            className="flex-1 cursor-pointer rounded-sm"
            style={{ background: STRIP_COLOR[f.classification] }}
          />
        ))}
      </div>

      <div className="mt-3">
        {result.suspicious_segments.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            {result.suspicious_segments.map((s, i) => (
              <div
                key={i}
                role="button"
                onClick={() => onSeek(s.start_timestamp)}
                className="flex cursor-pointer items-center justify-between rounded-lg bg-[var(--color-surface-raised)] px-3 py-2 text-sm hover:outline hover:outline-1 hover:outline-[var(--color-accent)]"
              >
                <span>
                  ⚠ {formatTimestamp(s.start_timestamp)} – {formatTimestamp(s.end_timestamp)} (
                  {s.frame_count} frames)
                </span>
                <span className="text-[var(--color-muted)]">peak {fakeScorePct(s.peak_fake_probability)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-[var(--color-muted)]">No sustained suspicious segments found.</div>
        )}
      </div>
    </div>
  );
}
