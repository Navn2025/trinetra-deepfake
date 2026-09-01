import { useEffect, useRef, useState } from "react";
import { Dropzone } from "../../components/ui/Dropzone";
import { StatusLine } from "../../components/ui/StatusLine";
import { Card } from "../../components/ui/Card";
import { StatTile } from "../../components/ui/StatTile";
import { ClassificationBadge } from "../../components/ui/Badge";
import { SecurityNote } from "../../components/ui/SecurityNote";
import { DownloadIcon } from "../../components/ui/Icons";
import { SettingRow } from "../../components/ui/Toggle";
import { Spinner } from "../../components/ui/Spinner";
import { fakeScorePct, formatTimestamp } from "../../lib/classification";
import { analyzeVideo } from "../../lib/api";
import { buildVideoReport, captureVideoFrame, computeSha256 } from "../../lib/report";
import {
  checkVoice,
  deleteEnrolledVoice,
  enrolledVoiceAudioUrl,
  enrollVoice,
  extractAudioAsWav,
  listEnrolledVoices,
  type VoiceAudioSegment,
  type VoiceCheckResult,
  type VoiceEnhancementInfo,
  type VoiceIdentityMatch,
} from "../../lib/voiceCheck";
import { ApiError } from "../../lib/types";
import type { AnalyzeVideoResult, Classification, FrameEvidence } from "../../lib/types";

const STRIP_COLOR: Record<string, string> = {
  real: "var(--color-real)",
  fake: "var(--color-fake)",
  uncertain: "var(--color-uncertain)",
  insufficient_quality: "var(--color-neutral)",
  no_face_detected: "var(--color-neutral)",
};

export function VideoAnalyzer() {
  const [voiceCheckEnabled, setVoiceCheckEnabled] = useState(true);
  const [inferFromVoice, setInferFromVoice] = useState(false);
  const [enhanceVoice, setEnhanceVoice] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeVideoResult | null>(null);
  const [downloadingReport, setDownloadingReport] = useState(false);

  const [voiceChecking, setVoiceChecking] = useState(false);
  const [voiceResult, setVoiceResult] = useState<VoiceCheckResult | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  const [enrolledNames, setEnrolledNames] = useState<string[]>([]);
  const [referenceName, setReferenceName] = useState<string>("");

  const videoRef = useRef<HTMLVideoElement>(null);

  function refreshEnrolledNames() {
    listEnrolledVoices()
      .then(setEnrolledNames)
      .catch(() => {
        // Voice-integrity service may not be up yet -- the enrollment
        // picker just stays empty until the next successful refresh
        // (re-enrolling or reopening this page), not worth surfacing as
        // an error on its own.
      });
  }

  useEffect(() => {
    refreshEnrolledNames();
  }, []);

  async function handleFile(f: File) {
    const url = URL.createObjectURL(f);
    setFile(f);
    setObjectUrl(url);
    setResult(null);
    setError(null);
    setUploading(true);
    setProgress(0);
    setProcessing(false);
    setVoiceResult(null);
    setVoiceError(null);
    setVoiceChecking(false);

    // Video analysis (backend :8000) and voice check (voice-integrity :8001)
    // are independent services -- fire both immediately instead of awaiting
    // one before starting the other, so wall-clock time is max(video,
    // voice) rather than their sum (each alone can take tens of seconds).
    const voicePromise = voiceCheckEnabled
      ? (async () => {
          setVoiceChecking(true);
          try {
            const wav = await extractAudioAsWav(f);
            if (!wav) {
              setVoiceError("No decodable audio track found in this file.");
            } else {
              setVoiceResult(await checkVoice(wav, enhanceVoice, referenceName || null));
            }
          } catch (err) {
            setVoiceError(err instanceof Error ? err.message : "Voice check failed");
          } finally {
            setVoiceChecking(false);
          }
        })()
      : Promise.resolve();

    try {
      const res = await analyzeVideo(f, (p) => {
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
      await voicePromise;
      return;
    }

    await voicePromise;
  }

  function seekTo(t: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      videoRef.current.play();
    }
  }

  // Escalation: a video the frame-by-frame model reads as real/uncertain
  // can still be a voice clone dubbed over real footage -- if the voice
  // check independently flags spoofing and the user opted into inferring
  // from it, the displayed verdict reflects that instead of silently
  // disagreeing with a result shown two lines above it.
  const escalated = inferFromVoice && voiceResult?.verdict === "FAKE/SPOOFED";
  const displayClassification: Classification | null = result
    ? escalated
      ? "fake"
      : result.overall_classification
    : null;

  async function handleDownloadReport() {
    if (!file || !result) return;
    setDownloadingReport(true);
    try {
      const checksum = await computeSha256(file);
      const mediaPreviewDataUrl = videoRef.current ? captureVideoFrame(videoRef.current) : null;
      const doc = buildVideoReport(result, {
        fileName: file.name,
        fileKind: "Video",
        fileSizeKB: file.size / 1024,
        submittedAt: new Date(),
        checksum,
        modelVersion: result.model_version,
        processingTimeMs: result.processing_time_ms,
        mediaPreviewDataUrl: mediaPreviewDataUrl ?? undefined,
        voiceResult,
        escalatedByVoice: escalated,
      });
      doc.save(`trinetra-report-${file.name.replace(/\.[^.]+$/, "")}.pdf`);
    } finally {
      setDownloadingReport(false);
    }
  }

  return (
    <div>
      <Card className="mb-4 p-2">
        <SettingRow
          title="Voice Check"
          description="After analyzing the video, extracts its audio track in-browser and runs it through a separate voice anti-spoofing model (Gustking + XLS-R/SLS) to check whether the speaker's voice is cloned/synthetic."
          checked={voiceCheckEnabled}
          onChange={setVoiceCheckEnabled}
          highlighted
        />
        <SettingRow
          title="Infer Detection from Voice"
          description="If Voice Check finds strong evidence of a cloned voice, the overall verdict shown is escalated to Fake, even if the frame-by-frame video model alone read the footage as real or uncertain."
          checked={inferFromVoice}
          onChange={setInferFromVoice}
          disabled={!voiceCheckEnabled}
          indented
        />
        <SettingRow
          title="Denoise & Enhance"
          description="Cleans the extracted audio (noise reduction + a neural speech enhancer) before scoring it. Can reduce false 'spoofed' flags caused by background noise or compression on genuine audio -- but can also smooth over the artifacts a real clone would show, so both the raw and enhanced scores are shown for comparison, not just the enhanced one."
          checked={enhanceVoice}
          onChange={setEnhanceVoice}
          disabled={!voiceCheckEnabled}
          indented
        />
      </Card>

      <VoiceEnrollmentCard
        enabled={voiceCheckEnabled}
        enrolledNames={enrolledNames}
        referenceName={referenceName}
        onReferenceNameChange={setReferenceName}
        onEnrolledNamesChange={refreshEnrolledNames}
      />

      <Dropzone
        accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
        title="Drop your video here"
        hint="Frames are sampled evenly across the clip"
        maxSizeLabel="Up to 200MB"
        onFile={handleFile}
      />
      <SecurityNote />

      {objectUrl && (
        <Card className="mt-4 p-4">
          <video ref={videoRef} src={objectUrl} controls className="block w-full rounded-lg" />

          <StatusLine uploading={uploading} processing={processing} progress={progress} error={error} />

          {result && displayClassification && (
            <ResultPanel
              result={result}
              displayClassification={displayClassification}
              escalated={escalated}
              onSeek={seekTo}
              onDownloadReport={handleDownloadReport}
              downloadingReport={downloadingReport}
              voiceCheckEnabled={voiceCheckEnabled}
              voiceChecking={voiceChecking}
              voiceResult={voiceResult}
              voiceError={voiceError}
            />
          )}
        </Card>
      )}
    </div>
  );
}

function ResultPanel({
  result,
  displayClassification,
  escalated,
  onSeek,
  onDownloadReport,
  downloadingReport,
  voiceCheckEnabled,
  voiceChecking,
  voiceResult,
  voiceError,
}: {
  result: AnalyzeVideoResult;
  displayClassification: Classification;
  escalated: boolean;
  onSeek: (t: number) => void;
  onDownloadReport: () => void;
  downloadingReport: boolean;
  voiceCheckEnabled: boolean;
  voiceChecking: boolean;
  voiceResult: VoiceCheckResult | null;
  voiceError: string | null;
}) {
  const m = result.metadata;
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ClassificationBadge classification={displayClassification} />
          <span className="text-xs text-[var(--color-muted)]">
            {fakeScorePct(result.overall_confidence)}
          </span>
          {escalated && (
            <span className="rounded bg-[var(--color-fake-bg)] px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-[var(--color-fake)] uppercase">
              Escalated by voice check
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="text-xs text-[var(--color-muted)]">
            {result.processing_time_ms}ms{result.model_version ? ` · ${result.model_version}` : ""}
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
                className="flex cursor-pointer items-center gap-3 rounded-lg bg-[var(--color-surface-raised)] px-3 py-2 text-sm hover:outline hover:outline-1 hover:outline-[var(--color-accent)]"
              >
                {s.peak_frame && (
                  <img
                    src={`data:image/jpeg;base64,${s.peak_frame.thumbnail}`}
                    alt=""
                    className="h-10 w-10 shrink-0 rounded object-cover"
                  />
                )}
                <div className="flex flex-1 flex-wrap items-center justify-between gap-1">
                  <span>
                    ⚠ {formatTimestamp(s.start_timestamp)} – {formatTimestamp(s.end_timestamp)} (
                    {s.frame_count} frames)
                  </span>
                  <span className="text-[var(--color-muted)]">peak {fakeScorePct(s.peak_fake_probability)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs text-[var(--color-muted)]">No sustained suspicious segments found.</div>
        )}
      </div>

      {result.most_suspicious_frame && (
        <FrameEvidenceCard
          title="Most suspicious frame"
          frame={result.most_suspicious_frame}
          onSeek={onSeek}
        />
      )}

      {voiceCheckEnabled && (
        <div className="mt-3 rounded-lg bg-[var(--color-surface-raised)] px-3 py-2.5 text-sm">
          {voiceChecking && (
            <div className="flex items-center gap-2 text-[var(--color-muted)]">
              <Spinner /> Running voice check…
            </div>
          )}
          {!voiceChecking && voiceError && (
            <div className="text-xs text-[var(--color-muted)]">🎙 Voice check: {voiceError}</div>
          )}
          {!voiceChecking && voiceResult && (
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-medium">
                  🎙{" "}
                  {voiceResult.verdict === "FAKE/SPOOFED" ? (
                    <span className="text-[var(--color-fake)]">Voice may be synthetic</span>
                  ) : (
                    <span className="text-[var(--color-real)]">Voice sounds genuine</span>
                  )}
                </span>
                <span className="text-xs text-[var(--color-muted)]">
                  xlsr_sls bonafide {Math.round(voiceResult.xlsr_sls.bonafide * 100)}% / spoof{" "}
                  {Math.round(voiceResult.xlsr_sls.spoof * 100)}% · gustking real{" "}
                  {Math.round((voiceResult.gustking.real ?? 0) * 100)}% / fake{" "}
                  {Math.round((voiceResult.gustking.fake ?? 0) * 100)}%
                </span>
              </div>
              {voiceResult.identity && <IdentityMatchCard identity={voiceResult.identity} />}
              {voiceResult.enhancement && <EnhancementComparison enhancement={voiceResult.enhancement} enhancedVerdict={voiceResult.verdict} />}
              {voiceResult.most_suspicious_segment && (
                <AudioSegmentEvidence segment={voiceResult.most_suspicious_segment} onSeek={onSeek} />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Evidence card for the single most suspicious sampled video frame: the
 * downsized full frame (not just the face crop) with a red box drawn over
 * the detected face, so it's visible *where* in the frame the model
 * reacted, not just a timestamp + number. */
function FrameEvidenceCard({
  title,
  frame,
  onSeek,
}: {
  title: string;
  frame: FrameEvidence;
  onSeek: (t: number) => void;
}) {
  return (
    <div className="mt-3 rounded-lg bg-[var(--color-surface-raised)] p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold tracking-wide text-[var(--color-muted)] uppercase">{title}</span>
        <span className="text-xs text-[var(--color-muted)]">
          {formatTimestamp(frame.timestamp)} · {fakeScorePct(frame.fake_probability)}
        </span>
      </div>
      <div className="relative inline-block max-w-full overflow-hidden rounded-md border border-[var(--color-border)]">
        <img
          src={`data:image/jpeg;base64,${frame.thumbnail}`}
          alt={`Sampled frame at ${formatTimestamp(frame.timestamp)}`}
          className="block max-h-72 max-w-full"
        />
        {frame.bbox && (
          <div
            className="pointer-events-none absolute border-2 border-[var(--color-fake)]"
            style={{
              left: `${frame.bbox.x * 100}%`,
              top: `${frame.bbox.y * 100}%`,
              width: `${frame.bbox.width * 100}%`,
              height: `${frame.bbox.height * 100}%`,
              boxShadow: "0 0 0 1px rgba(0,0,0,0.45)",
            }}
          />
        )}
      </div>
      <button
        onClick={() => onSeek(frame.timestamp)}
        className="mt-2 text-xs font-medium text-[var(--color-accent)] hover:underline"
      >
        Jump to this moment →
      </button>
    </div>
  );
}

/** Equivalent evidence for audio: no image to box, so instead this surfaces
 * *which time window* of the clip the anti-spoofing models reacted to
 * most, with a button to seek+play that exact moment (the same <video>
 * element carries the audio track, so onSeek doubles as "play this part"). */
function AudioSegmentEvidence({
  segment,
  onSeek,
}: {
  segment: VoiceAudioSegment;
  onSeek: (t: number) => void;
}) {
  return (
    <div className="mt-2 rounded-lg bg-[var(--color-surface)] px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold tracking-wide uppercase">
          {segment.flagged ? (
            <span className="text-[var(--color-fake)]">⚠ Most suspicious moment</span>
          ) : (
            <span className="text-[var(--color-muted)]">Most suspicious moment</span>
          )}
        </span>
        <button
          onClick={() => onSeek(segment.start)}
          className="text-xs font-medium text-[var(--color-accent)] hover:underline"
        >
          ▶ Play this part
        </button>
      </div>
      <div className="mt-1 text-xs text-[var(--color-muted)]">
        {formatTimestamp(segment.start)} – {formatTimestamp(segment.end)} · xlsr_sls bonafide{" "}
        {Math.round(segment.xlsr_sls.bonafide * 100)}% / spoof {Math.round(segment.xlsr_sls.spoof * 100)}% ·
        gustking real {Math.round((segment.gustking.real ?? 0) * 100)}% / fake{" "}
        {Math.round((segment.gustking.fake ?? 0) * 100)}%
      </div>
    </div>
  );
}

/** Lets the user enroll a reference voice sample (name + audio file) for
 * speaker-identity matching, and pick which enrolled person the next voice
 * check should be compared against. Purely additive to the existing spoof
 * check -- picking "None" (the default) runs voice check exactly as
 * before. */
function VoiceEnrollmentCard({
  enabled,
  enrolledNames,
  referenceName,
  onReferenceNameChange,
  onEnrolledNamesChange,
}: {
  enabled: boolean;
  enrolledNames: string[];
  referenceName: string;
  onReferenceNameChange: (name: string) => void;
  onEnrolledNamesChange: () => void;
}) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [enrolling, setEnrolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleEnroll() {
    if (!name.trim() || !file) return;
    setEnrolling(true);
    setError(null);
    try {
      // The enrolled sample is read by speechbrain via `soundfile`, which
      // (like the two anti-spoofing models) only reads raw audio
      // containers -- not video (mp4/webm/mov). Video files must go
      // through the same client-side audio extraction the main voice
      // check already uses, or the server gets a video container it can't
      // open ("Format not recognised").
      const isVideo = file.type.startsWith("video/");
      const uploadBlob = isVideo ? await extractAudioAsWav(file) : file;
      if (!uploadBlob) {
        setError("No decodable audio track found in this file.");
        return;
      }
      const { name: savedName } = await enrollVoice(name.trim(), uploadBlob, isVideo ? "reference.wav" : file.name);
      onEnrolledNamesChange();
      onReferenceNameChange(savedName);
      setName("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Enrollment failed");
    } finally {
      setEnrolling(false);
    }
  }

  async function handleDelete(target: string) {
    try {
      await deleteEnrolledVoice(target);
      if (referenceName === target) onReferenceNameChange("");
      onEnrolledNamesChange();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <Card className="mb-4 p-3 text-sm">
      <div className="font-medium">🪪 Voice identity match (optional)</div>
      <div className="mt-1 text-xs text-[var(--color-muted)]">
        Enroll a known-genuine sample of someone's voice, then check whether the video's voice matches them. This
        is separate from the spoof check above -- a convincing clone of the enrolled voice will legitimately show
        as a match, so use this to confirm identity, not authenticity.
      </div>

      {enrolledNames.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <label className="text-xs text-[var(--color-muted)]">Match video's voice against:</label>
          <select
            value={referenceName}
            onChange={(e) => onReferenceNameChange(e.target.value)}
            disabled={!enabled}
            className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-raised)] px-2 py-1 text-xs disabled:opacity-50"
          >
            <option value="">None</option>
            {enrolledNames.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          {referenceName && (
            <button
              onClick={() => handleDelete(referenceName)}
              className="text-xs text-[var(--color-muted)] hover:text-[var(--color-fake)] hover:underline"
            >
              Remove "{referenceName}"
            </button>
          )}
        </div>
      )}

      {referenceName && (
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs text-[var(--color-muted)]">Listen to "{referenceName}"'s enrolled sample:</span>
          {/* key forces the element to reload its source when the selected
              reference changes, instead of silently keeping the previous
              enrollment's audio -- exactly the kind of "trusting the name
              instead of verifying by ear" mixup this player exists to catch. */}
          <audio key={referenceName} controls src={enrolledVoiceAudioUrl(referenceName)} className="h-8 max-w-64" />
        </div>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Person's name"
          className="w-40 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-raised)] px-2 py-1 text-xs"
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,video/mp4,video/webm,video/quicktime"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="max-w-56 text-xs"
        />
        <button
          onClick={handleEnroll}
          disabled={!name.trim() || !file || enrolling}
          className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] px-3 py-1.5 text-xs font-medium transition-colors hover:bg-[var(--color-border)] disabled:opacity-50"
        >
          {enrolling ? "Enrolling…" : "Enroll voice"}
        </button>
      </div>
      {error && <div className="mt-1 text-xs text-[var(--color-fake)]">{error}</div>}
    </Card>
  );
}

// speechbrain's SpeakerRecognition.verify_files() default same-speaker cutoff
// (venv/Lib/site-packages/speechbrain/inference/speaker.py) -- a cosine
// similarity between speaker embeddings, NOT a classifier probability like
// the 0-100% scores elsewhere in this panel. Genuine same-speaker pairs
// routinely land well under 1.0 (different recording, day, device,
// background noise all pull it down), so a bare "0.663" reads as a weak
// score next to 94%/100% figures when it's actually ~2.6x the match
// threshold -- shown qualitatively here instead of as a lone raw number.
const SAME_SPEAKER_THRESHOLD = 0.25;

/** Result of matching the analyzed video's voice against an enrolled
 * reference -- identity only, independent of the spoof verdict shown
 * alongside it (see VoiceEnrollmentCard's description). */
function IdentityMatchCard({ identity }: { identity: VoiceIdentityMatch }) {
  const confidence = identity.same_speaker
    ? identity.similarity >= SAME_SPEAKER_THRESHOLD * 2
      ? "Strong match"
      : "Match"
    : "No match";
  return (
    <div className="mt-2 rounded-lg bg-[var(--color-surface)] px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold tracking-wide uppercase">
          {identity.same_speaker ? (
            <span className="text-[var(--color-real)]">
              🪪 Matches "{identity.reference_name}" ({confidence})
            </span>
          ) : (
            <span className="text-[var(--color-fake)]">🪪 Does not match "{identity.reference_name}"</span>
          )}
        </span>
        <span className="text-xs text-[var(--color-muted)]">
          similarity {identity.similarity.toFixed(3)} (match threshold {SAME_SPEAKER_THRESHOLD})
        </span>
      </div>
      <div className="mt-1 text-[10px] text-[var(--color-muted)]">
        Cosine similarity between voice embeddings, not a percentage confidence -- values well under 1.0 are normal
        for a genuine match.
      </div>
    </div>
  );
}

/** Shows the pre-enhancement ("raw") scores next to the post-enhancement
 * ones the rest of the panel is already displaying, so "Denoise & Enhance"
 * never hides what the unmodified audio actually scored -- see
 * audio_enhance.py's module docstring for why enhancement is a second
 * opinion, not a strict improvement. Flags loudly when it changed the
 * verdict, since that's the case worth a second look either way. */
function EnhancementComparison({
  enhancement,
  enhancedVerdict,
}: {
  enhancement: VoiceEnhancementInfo;
  enhancedVerdict: "FAKE/SPOOFED" | "LIKELY GENUINE";
}) {
  const verdictChanged = enhancement.raw.verdict !== enhancedVerdict;
  return (
    <div className="mt-2 rounded-lg bg-[var(--color-surface)] px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold tracking-wide text-[var(--color-muted)] uppercase">
          🧼 Denoise &amp; Enhance applied
        </span>
        {verdictChanged && (
          <span className="rounded bg-[var(--color-uncertain-bg)] px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-[var(--color-uncertain)] uppercase">
            Verdict changed
          </span>
        )}
      </div>
      <div className="mt-1 text-xs text-[var(--color-muted)]">
        Raw ({enhancement.raw.verdict === "FAKE/SPOOFED" ? "may be synthetic" : "sounds genuine"}): xlsr_sls
        spoof {Math.round(enhancement.raw.xlsr_sls.spoof * 100)}% · gustking fake{" "}
        {Math.round((enhancement.raw.gustking.fake ?? 0) * 100)}%
      </div>
      <div className="mt-2 text-[10px] text-[var(--color-muted)]">
        {enhancement.denoiser} + {enhancement.enhancer}. The scores above (outside this box) are the enhanced
        clip's; this box is the same clip before enhancement, for comparison.
      </div>
    </div>
  );
}
