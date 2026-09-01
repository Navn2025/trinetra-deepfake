// PDF report generation, client-side only (no backend changes needed).
//
// Modeled on Resemble.AI's "Detect Report" layout (header + summary cards +
// key/value detail tables + media preview + a fakeness-over-time timeline
// for video), but only includes sections this project can back with real
// data. The reference report also has an "Intelligence Results" section
// (LLM-generated speaker/context/fraud/liveness analysis from a panel of
// "AI experts") and a C2PA content-credentials check -- neither exists in
// this project, so neither appears here. Fabricating that kind of
// analysis would be actively misleading in a report whose whole purpose
// is telling someone whether media is trustworthy.
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type {
  AnalyzeImageResult,
  AnalyzeVideoResult,
  Classification,
  HistoryEntry,
} from "./types";
import type { VoiceCheckResult } from "./voiceCheck";
import { CLASSIFICATION_LABEL, classifyFakeProbability, formatDate, parseSqliteUtc } from "./classification";

const PAGE_MARGIN = 14;
const PAGE_WIDTH = 210; // A4, mm

const INK = [35, 33, 29] as const;
const MUTED = [81, 76, 66] as const;
const BORDER = [214, 203, 172] as const;
const REAL = [63, 125, 83] as const;
const FAKE = [168, 56, 42] as const;
const UNCERTAIN = [156, 122, 46] as const;
const NEUTRAL = [139, 130, 114] as const;

function verdictColor(c: Classification): readonly [number, number, number] {
  if (c === "real") return REAL;
  if (c === "fake") return FAKE;
  if (c === "uncertain") return UNCERTAIN;
  return NEUTRAL;
}

function verdictLabel(c: Classification): string {
  if (c === "fake") return "Deepfake Detected";
  if (c === "real") return "No Deepfake Detected";
  if (c === "uncertain") return "Uncertain";
  return CLASSIFICATION_LABEL[c];
}

export interface ReportMeta {
  fileName: string;
  fileKind: "Image" | "Video" | "Live Capture";
  fileSizeKB: number | null;
  submittedAt: Date;
  checksum?: string; // SHA-256 hex -- see computeSha256 below. Not MD5: browsers'
  // native crypto.subtle doesn't expose MD5 (deprecated/insecure), so this
  // uses SHA-256 and labels it as such rather than mislabeling the algorithm.
  modelVersion?: string | null;
  processingTimeMs?: number;
  mediaPreviewDataUrl?: string; // image/jpeg or image/png data URL
  // Video reports only -- result of the optional client-side "Voice Check"
  // (see VideoAnalyzer.tsx/voiceCheck.ts). null/undefined if the option was
  // off or the file had no decodable audio track.
  voiceResult?: VoiceCheckResult | null;
  escalatedByVoice?: boolean;
}

/** SHA-256 of a File's contents, hex-encoded -- via the browser's native
 * SubtleCrypto, no dependency needed. */
export async function computeSha256(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Captures the current frame of a <video> element as a JPEG data URL --
 * used for the video report's media preview, since analyze-video doesn't
 * return a server-side thumbnail. */
export function captureVideoFrame(video: HTMLVideoElement): string | null {
  if (!video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function drawHeader(doc: jsPDF, title: string, subtitle: string): number {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(...FAKE);
  doc.text("TRINETRA · DEEPFAKE DETECTION REPORT", PAGE_MARGIN, 16);

  doc.setFontSize(16);
  doc.setTextColor(...INK);
  doc.text(title, PAGE_MARGIN, 25);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text(subtitle, PAGE_MARGIN, 31);

  doc.setDrawColor(...BORDER);
  doc.line(PAGE_MARGIN, 35, PAGE_WIDTH - PAGE_MARGIN, 35);

  return 42;
}

function drawSummaryCards(
  doc: jsPDF,
  y: number,
  cards: { label: string; value: string; color?: readonly [number, number, number] }[],
): number {
  const cardWidth = (PAGE_WIDTH - PAGE_MARGIN * 2 - (cards.length - 1) * 4) / cards.length;
  const cardHeight = 20;

  cards.forEach((card, i) => {
    const x = PAGE_MARGIN + i * (cardWidth + 4);
    doc.setDrawColor(...BORDER);
    doc.setFillColor(246, 241, 228);
    doc.roundedRect(x, y, cardWidth, cardHeight, 1.5, 1.5, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(6.5);
    doc.setTextColor(...MUTED);
    doc.text(card.label.toUpperCase(), x + 3, y + 6);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(...(card.color ?? INK));
    const valueLines = doc.splitTextToSize(card.value, cardWidth - 6);
    doc.text(valueLines, x + 3, y + 13);
  });

  return y + cardHeight + 10;
}

function drawTable(doc: jsPDF, y: number, title: string, rows: [string, string][]): number {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...INK);
  doc.text(title, PAGE_MARGIN, y);

  autoTable(doc, {
    startY: y + 3,
    margin: { left: PAGE_MARGIN, right: PAGE_MARGIN },
    body: rows,
    theme: "plain",
    styles: {
      fontSize: 9,
      textColor: INK as unknown as [number, number, number],
      cellPadding: { top: 2.5, bottom: 2.5, left: 0, right: 3 },
      lineColor: BORDER as unknown as [number, number, number],
      lineWidth: 0.1,
    },
    columnStyles: {
      0: { fontStyle: "bold", textColor: MUTED as unknown as [number, number, number], cellWidth: 55 },
    },
    didParseCell: (data) => {
      data.cell.styles.lineWidth = { top: 0, right: 0, bottom: 0.1, left: 0 };
    },
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (doc as any).lastAutoTable.finalY + 10;
}

function drawMediaPreview(doc: jsPDF, y: number, dataUrl: string, caption: string): number {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...INK);
  doc.text("Media Preview", PAGE_MARGIN, y);
  y += 4;

  const maxWidth = PAGE_WIDTH - PAGE_MARGIN * 2;
  const maxHeight = 70;

  const img = new Image();
  img.src = dataUrl;
  const aspect = img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 1;
  let w = maxWidth;
  let h = w / aspect;
  if (h > maxHeight) {
    h = maxHeight;
    w = h * aspect;
  }

  doc.setDrawColor(...BORDER);
  doc.roundedRect(PAGE_MARGIN, y, w, h, 1.5, 1.5);
  doc.addImage(dataUrl, "JPEG", PAGE_MARGIN, y, w, h, undefined, "FAST");
  y += h + 4;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(...MUTED);
  doc.text(caption, PAGE_MARGIN, y);

  return y + 8;
}

/** Renders a fakeness-over-time line chart to a PNG data URL via an
 * offscreen canvas -- no charting library dependency needed for one chart. */
function renderTimelineChart(frames: { timestamp: number; fake_probability: number | null }[]): string | null {
  const scored = frames.filter((f) => f.fake_probability !== null) as { timestamp: number; fake_probability: number }[];
  if (scored.length === 0) return null;

  const width = 900;
  const height = 260;
  const padding = { top: 16, right: 16, bottom: 30, left: 46 };
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const maxT = Math.max(...scored.map((f) => f.timestamp), 1);

  // gridlines + axis labels (0/25/50/75/100%)
  ctx.strokeStyle = "#e5ded0";
  ctx.fillStyle = "#514c42";
  ctx.font = "12px Arial";
  ctx.textAlign = "right";
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = padding.top + plotH * (1 - pct / 100);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(`${pct}%`, padding.left - 6, y + 4);
  }

  // fake-threshold reference line (0.65, matches config.py's FAKE_THRESHOLD)
  const thresholdY = padding.top + plotH * (1 - 0.65);
  ctx.strokeStyle = "#a8382a55";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(padding.left, thresholdY);
  ctx.lineTo(width - padding.right, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);

  // the line itself
  ctx.strokeStyle = "#a8382a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  scored.forEach((f, i) => {
    const x = padding.left + (f.timestamp / maxT) * plotW;
    const y = padding.top + plotH * (1 - f.fake_probability);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // x-axis labels
  ctx.fillStyle = "#514c42";
  ctx.textAlign = "center";
  const steps = 6;
  for (let i = 0; i <= steps; i++) {
    const t = (maxT / steps) * i;
    const x = padding.left + (t / maxT) * plotW;
    ctx.fillText(`${t.toFixed(1)}s`, x, height - padding.bottom + 16);
  }

  return canvas.toDataURL("image/png");
}

function drawTimelineChart(doc: jsPDF, y: number, frames: { timestamp: number; fake_probability: number | null }[]): number {
  const chart = renderTimelineChart(frames);
  if (!chart) return y;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...INK);
  doc.text("Fake Score Over Time", PAGE_MARGIN, y);
  y += 4;

  const w = PAGE_WIDTH - PAGE_MARGIN * 2;
  const h = (w * 260) / 900;
  doc.setDrawColor(...BORDER);
  doc.roundedRect(PAGE_MARGIN, y, w, h, 1.5, 1.5);
  doc.addImage(chart, "PNG", PAGE_MARGIN, y, w, h, undefined, "FAST");
  y += h + 4;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...MUTED);
  doc.text("Dashed line marks the fake-classification threshold (65%).", PAGE_MARGIN, y);

  return y + 8;
}

function drawFooter(doc: jsPDF) {
  const pageHeight = doc.internal.pageSize.getHeight();
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...MUTED);
  doc.text(
    "Automated analysis from a research-grade model, not a certified forensic determination. Generated by Trinetra.",
    PAGE_MARGIN,
    pageHeight - 10,
  );
  doc.text(`Generated ${new Date().toISOString()}`, PAGE_MARGIN, pageHeight - 6);
}

export function buildImageReport(result: AnalyzeImageResult, meta: ReportMeta): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const classification = result.overall_classification;
  const color = verdictColor(classification);

  let y = drawHeader(doc, meta.fileName, `Image · ${formatDateFull(meta.submittedAt)}`);

  y = drawSummaryCards(doc, y, [
    { label: "Verdict", value: verdictLabel(classification), color },
    { label: "Faces Detected", value: String(result.faces.length) },
    { label: "Submitted", value: formatDateFull(meta.submittedAt) },
    {
      label: "Confidence",
      value: result.overall_confidence === null ? "—" : `${Math.round(result.overall_confidence * 100)}%`,
      color,
    },
  ]);

  y = drawTable(doc, y, "File Details", [
    ["File Type", meta.fileKind],
    ["File Name", meta.fileName],
    ["Status", verdictLabel(classification)],
    ["Date Submitted", formatDateFull(meta.submittedAt)],
    ["Size (KB)", meta.fileSizeKB !== null ? meta.fileSizeKB.toFixed(2) : "—"],
    ["Checksum (SHA-256)", meta.checksum ?? "—"],
    ["Model Version", meta.modelVersion ?? "—"],
    ["Processing Time", meta.processingTimeMs !== undefined ? `${meta.processingTimeMs}ms` : "—"],
  ]);

  y = drawTable(doc, y, "Detection Results", [
    ["Overall Classification", CLASSIFICATION_LABEL[classification]],
    [
      "Overall Confidence",
      result.overall_confidence === null ? "—" : `${(result.overall_confidence * 100).toFixed(2)}%`,
    ],
  ]);

  if (result.faces.length > 0) {
    if (y > 240) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    y = drawTable(
      doc,
      y,
      "Per-Face Breakdown",
      result.faces.map((f, i) => [
        `Face ${i + 1}`,
        `${CLASSIFICATION_LABEL[f.classification]} · ${
          f.fake_probability === null ? "—" : `${(f.fake_probability * 100).toFixed(1)}% fake score`
        }${f.quality_issue ? ` · ${f.quality_issue}` : ""}`,
      ]),
    );
  }

  if (meta.mediaPreviewDataUrl) {
    if (y > 200) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    drawMediaPreview(doc, y, meta.mediaPreviewDataUrl, meta.fileName);
  }

  drawFooter(doc);
  return doc;
}

export function buildVideoReport(result: AnalyzeVideoResult, meta: ReportMeta): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  // Escalated: the frame-by-frame video model's own verdict is left
  // unchanged in "Detection Results" below (that's what it actually
  // found), but the headline verdict reflects the voice check too, same as
  // the in-app display -- see VideoAnalyzer.tsx's `escalated` logic.
  const classification = meta.escalatedByVoice ? "fake" : result.overall_classification;
  const color = verdictColor(classification);
  const m = result.metadata;

  let y = drawHeader(doc, meta.fileName, `Video · ${formatDateFull(meta.submittedAt)}`);

  y = drawSummaryCards(doc, y, [
    {
      label: "Verdict",
      value: meta.escalatedByVoice ? "Deepfake Detected (voice)" : verdictLabel(classification),
      color,
    },
    { label: "Duration", value: `${m.duration_seconds}s` },
    { label: "Submitted", value: formatDateFull(meta.submittedAt) },
    {
      label: "Confidence",
      value: result.overall_confidence === null ? "—" : `${Math.round(result.overall_confidence * 100)}%`,
      color,
    },
  ]);

  y = drawTable(doc, y, "File Details", [
    ["File Type", meta.fileKind],
    ["File Name", meta.fileName],
    ["Status", meta.escalatedByVoice ? "Deepfake Detected (escalated by voice check)" : verdictLabel(classification)],
    ["Date Submitted", formatDateFull(meta.submittedAt)],
    ["Size (KB)", meta.fileSizeKB !== null ? meta.fileSizeKB.toFixed(2) : "—"],
    ["Checksum (SHA-256)", meta.checksum ?? "—"],
    ["Model Version", meta.modelVersion ?? "—"],
    ["Processing Time", meta.processingTimeMs !== undefined ? `${meta.processingTimeMs}ms` : "—"],
  ]);

  y = drawTable(doc, y, "Video Details", [
    ["Duration", `${m.duration_seconds}s`],
    ["Native FPS", String(m.fps)],
    ["Resolution", `${m.width}×${m.height}`],
    ["Total Frames", String(m.total_frames)],
    ["Frames Analyzed", String(m.frames_analyzed)],
  ]);

  y = drawTable(doc, y, "Detection Results (video, frame-by-frame)", [
    ["Overall Classification", CLASSIFICATION_LABEL[result.overall_classification]],
    [
      "Overall Confidence",
      result.overall_confidence === null ? "—" : `${(result.overall_confidence * 100).toFixed(2)}%`,
    ],
    ["Suspicious Segments", String(result.suspicious_segments.length)],
  ]);

  if (meta.voiceResult) {
    if (y > 240) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    const v = meta.voiceResult;
    const gustkingFake = v.gustking.fake ?? Math.max(...Object.values(v.gustking));
    const voiceRows: [string, string][] = [
      ["Verdict", v.verdict === "FAKE/SPOOFED" ? "Voice May Be Synthetic" : "Voice Sounds Genuine"],
      ["Gustking (wav2vec2-XLSR) Fake Score", `${(gustkingFake * 100).toFixed(2)}%`],
      ["XLS-R+SLS Bonafide / Spoof", `${(v.xlsr_sls.bonafide * 100).toFixed(2)}% / ${(v.xlsr_sls.spoof * 100).toFixed(2)}%`],
      ["Used to Escalate Overall Verdict", meta.escalatedByVoice ? "Yes" : "No"],
    ];
    if (v.identity) {
      voiceRows.push([
        "Voice Identity Match",
        `${v.identity.same_speaker ? "Matches" : "Does not match"} "${v.identity.reference_name}" (similarity ${v.identity.similarity.toFixed(3)})`,
      ]);
    }
    if (v.enhancement) {
      voiceRows.push([
        "Denoise & Enhance",
        `Applied (${v.enhancement.denoiser} + ${v.enhancement.enhancer}) -- pre-enhancement: ${
          v.enhancement.raw.verdict === "FAKE/SPOOFED" ? "May Be Synthetic" : "Sounds Genuine"
        }, spoof ${(v.enhancement.raw.xlsr_sls.spoof * 100).toFixed(1)}%`,
      ]);
    }
    if (v.most_suspicious_segment) {
      const seg = v.most_suspicious_segment;
      voiceRows.push([
        "Most Suspicious Audio Moment",
        `${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s · spoof ${(seg.xlsr_sls.spoof * 100).toFixed(1)}%${seg.flagged ? " (flagged)" : ""}`,
      ]);
    }
    y = drawTable(doc, y, "Voice Check Results (audio, independent model)", voiceRows);
  }

  if (y > 220) {
    doc.addPage();
    y = PAGE_MARGIN;
  }
  y = drawTimelineChart(doc, y, result.frames);

  if (result.suspicious_segments.length > 0) {
    if (y > 230) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    y = drawTable(
      doc,
      y,
      "Suspicious Segments",
      result.suspicious_segments.map((s, i) => [
        `Segment ${i + 1}`,
        `${s.start_timestamp.toFixed(1)}s – ${s.end_timestamp.toFixed(1)}s · peak ${(s.peak_fake_probability * 100).toFixed(1)}% · ${s.frame_count} frame(s)`,
      ]),
    );
  }

  if (result.most_suspicious_frame) {
    if (y > 200) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    const f = result.most_suspicious_frame;
    y = drawMediaPreview(
      doc,
      y,
      `data:image/jpeg;base64,${f.thumbnail}`,
      `Most suspicious sampled frame · ${f.timestamp.toFixed(1)}s · ${
        f.fake_probability === null ? "—" : `${(f.fake_probability * 100).toFixed(1)}% fake score`
      }`,
    );
  }

  if (meta.mediaPreviewDataUrl) {
    if (y > 200) {
      doc.addPage();
      y = PAGE_MARGIN;
    }
    drawMediaPreview(doc, y, meta.mediaPreviewDataUrl, meta.fileName);
  }

  drawFooter(doc);
  return doc;
}

/** Compact report for a /history entry (real-time /predict capture) --
 * these only ever carry a single-crop result (platform, fake_probability,
 * thumbnail, optional identity/scam fields), not a full analyze-image or
 * analyze-video result, so this deliberately has fewer sections than the
 * two builders above rather than padding with fields that don't exist. */
export function buildHistoryReport(entry: HistoryEntry): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const classification = classifyFakeProbability(entry.fake_probability);
  const color = verdictColor(classification);
  const submittedAt = parseSqliteUtc(entry.created_at);

  let y = drawHeader(doc, "Live Capture Detection Report", `${entry.platform ?? "Unknown platform"} · ${formatDate(entry.created_at)}`);

  y = drawSummaryCards(doc, y, [
    { label: "Verdict", value: verdictLabel(classification), color },
    { label: "Platform", value: entry.platform ?? "—" },
    { label: "Captured", value: formatDateFull(submittedAt) },
    { label: "Fake Score", value: `${Math.round(entry.fake_probability * 100)}%`, color },
  ]);

  const rows: [string, string][] = [
    ["Platform", entry.platform ?? "—"],
    ["Captured At", formatDateFull(submittedAt)],
    ["Classification", CLASSIFICATION_LABEL[classification]],
    ["Fake Score", `${(entry.fake_probability * 100).toFixed(2)}%`],
  ];
  if (entry.identity_contact_name) {
    rows.push([
      "Identity Match",
      `${entry.identity_contact_name} (${Math.round((entry.identity_similarity ?? 0) * 100)}% similarity)`,
    ]);
  }
  if (entry.scam_likelihood !== null) {
    rows.push(["Scam Likelihood", `${Math.round(entry.scam_likelihood * 100)}%`]);
  }
  y = drawTable(doc, y, "Detection Details", rows);

  drawMediaPreview(doc, y, `data:image/jpeg;base64,${entry.thumbnail_b64}`, "Captured frame thumbnail");

  drawFooter(doc);
  return doc;
}

function formatDateFull(d: Date): string {
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
