import { useEffect, useRef, useState } from "react";
import { Dropzone } from "../../components/ui/Dropzone";
import { StatusLine } from "../../components/ui/StatusLine";
import { Card } from "../../components/ui/Card";
import { ClassificationBadge } from "../../components/ui/Badge";
import { SecurityNote } from "../../components/ui/SecurityNote";
import { DownloadIcon } from "../../components/ui/Icons";
import { fakeScorePct } from "../../lib/classification";
import { analyzeImage } from "../../lib/api";
import { buildImageReport, computeSha256 } from "../../lib/report";
import { ApiError } from "../../lib/types";
import type { AnalyzeImageResult } from "../../lib/types";

const BOX_COLOR: Record<string, string> = {
  real: "#3f7d53",
  fake: "#a8382a",
  uncertain: "#9c7a2e",
  insufficient_quality: "#8b8272",
  no_face_detected: "#8b8272",
};

export function ImageAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeImageResult | null>(null);
  const [downloadingReport, setDownloadingReport] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => () => { if (objectUrl) URL.revokeObjectURL(objectUrl); }, [objectUrl]);

  function drawBoxes(current: AnalyzeImageResult) {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.naturalWidth) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.font = "bold 13px Inter, sans-serif";

    for (const f of current.faces) {
      const x = f.bbox.x * canvas.width;
      const y = f.bbox.y * canvas.height;
      const w = f.bbox.width * canvas.width;
      const h = f.bbox.height * canvas.height;
      const color = BOX_COLOR[f.classification] ?? "#6b7080";
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      const label = `${f.classification} ${fakeScorePct(f.fake_probability)}`;
      const textW = ctx.measureText(label).width + 10;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 20), textW, 20);
      ctx.fillStyle = "white";
      ctx.fillText(label, x + 5, Math.max(14, y - 5));
    }
  }

  async function handleFile(f: File) {
    const url = URL.createObjectURL(f);
    setFile(f);
    setObjectUrl(url);
    setResult(null);
    setError(null);
    setUploading(true);
    setProgress(0);
    setProcessing(false);

    try {
      const res = await analyzeImage(f, (p) => setProgress(p));
      setUploading(false);
      setProcessing(false);
      setResult(res);
    } catch (err) {
      setUploading(false);
      setProcessing(false);
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    }
  }

  async function handleDownloadReport() {
    if (!file || !result) return;
    setDownloadingReport(true);
    try {
      const [checksum, mediaPreviewDataUrl] = await Promise.all([
        computeSha256(file),
        Promise.resolve(canvasToImageDataUrl(imgRef.current)),
      ]);
      const doc = buildImageReport(result, {
        fileName: file.name,
        fileKind: "Image",
        fileSizeKB: file.size / 1024,
        submittedAt: new Date(),
        checksum,
        modelVersion: result.model_version,
        processingTimeMs: result.processing_time_ms,
        mediaPreviewDataUrl: mediaPreviewDataUrl ?? undefined,
      });
      doc.save(`trinetra-report-${file.name.replace(/\.[^.]+$/, "")}.pdf`);
    } finally {
      setDownloadingReport(false);
    }
  }

  return (
    <div>
      <Dropzone
        accept="image/jpeg,image/png,image/webp"
        title="Drop your image here"
        hint="Every face in the photo is analyzed separately"
        maxSizeLabel="Up to 8MB"
        onFile={handleFile}
      />
      <SecurityNote />

      {objectUrl && (
        <Card className="mt-4 p-4">
          <div className="relative inline-block max-w-full">
            <img
              ref={imgRef}
              src={objectUrl}
              className="block max-w-full rounded-lg"
              onLoad={() => result && drawBoxes(result)}
            />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          </div>

          <StatusLine uploading={uploading} processing={processing} progress={progress} error={error} />

          {result && (
            <ResultPanel
              result={result}
              onLayout={() => drawBoxes(result)}
              onDownloadReport={handleDownloadReport}
              downloadingReport={downloadingReport}
            />
          )}
        </Card>
      )}
    </div>
  );
}

/** Snapshots a loaded <img> element into a JPEG data URL for embedding in
 * the PDF report -- the objectUrl itself is a blob: URL, which jsPDF's
 * addImage can't load directly. */
function canvasToImageDataUrl(img: HTMLImageElement | null): string | null {
  if (!img || !img.naturalWidth) return null;
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(img, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function ResultPanel({
  result,
  onLayout,
  onDownloadReport,
  downloadingReport,
}: {
  result: AnalyzeImageResult;
  onLayout: () => void;
  onDownloadReport: () => void;
  downloadingReport: boolean;
}) {
  useEffect(() => {
    onLayout();
    window.addEventListener("resize", onLayout);
    return () => window.removeEventListener("resize", onLayout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ClassificationBadge classification={result.overall_classification} />
          <span className="text-xs text-[var(--color-muted)]">
            {fakeScorePct(result.overall_confidence)}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-xs text-[var(--color-muted)]">
            {result.faces.length} face(s) detected · {result.processing_time_ms}ms
            {result.model_version ? ` · ${result.model_version}` : ""}
          </div>
          <button
            onClick={onDownloadReport}
            disabled={downloadingReport}
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] px-3 py-1.5 text-xs font-medium text-[var(--color-text)] transition-colors hover:bg-[var(--color-border)] disabled:opacity-50"
          >
            <DownloadIcon width={14} height={14} />
            {downloadingReport ? "Generating…" : "Download Report"}
          </button>
        </div>
      </div>

      {result.faces.length > 0 && (
        <div className="mt-3 flex flex-col gap-1.5">
          {result.faces.map((f, i) => (
            <div key={i}>
              <div className="flex items-center justify-between rounded-lg bg-[var(--color-surface-raised)] px-3 py-2 text-sm">
                <span>Face {i + 1} (detection confidence {Math.round(f.detection_confidence * 100)}%)</span>
                <span className="flex items-center gap-2">
                  <ClassificationBadge classification={f.classification} />
                  <span className="text-xs text-[var(--color-muted)]">{fakeScorePct(f.fake_probability)}</span>
                </span>
              </div>
              {f.quality_issue && (
                <div className="mt-1 px-1 text-xs text-[var(--color-uncertain)]">{f.quality_issue}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
