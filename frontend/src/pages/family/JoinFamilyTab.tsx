import { useRef, useState } from "react";
import { Card } from "../../components/ui/Card";
import { Spinner } from "../../components/ui/Spinner";
import { joinFamily } from "../../lib/api";
import { ApiError } from "../../lib/types";

export function JoinFamilyTab() {
  const [familyId, setFamilyId] = useState("");
  const [passcode, setPasscode] = useState("");
  const [name, setName] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [thumbs, setThumbs] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const photosRef = useRef<Blob[]>([]);

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      setError("Couldn't access the camera — check browser permissions.");
    }
  }

  async function capturePhotos() {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    photosRef.current = [];
    setThumbs([]);

    for (let i = 0; i < 3; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      if (blob) {
        photosRef.current.push(blob);
        setThumbs((prev) => [...prev, URL.createObjectURL(blob)]);
      }
    }
    setCapturing(false);
  }

  async function handleSubmit() {
    if (!familyId || !passcode || !name.trim()) {
      setError("Fill in family ID, passcode, and your name.");
      return;
    }
    if (photosRef.current.length !== 3) {
      setError("Capture 3 face photos first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await joinFamily(Number(familyId), name.trim(), passcode, photosRef.current);
      setNotice(
        `Request submitted as "${result.name}" (id ${result.id}). Waiting for the head of family to approve.`,
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to submit join request");
    } finally {
      setSubmitting(false);
    }
  }

  const ready = thumbs.length === 3;

  return (
    <Card className="max-w-md p-5">
      <h2 className="text-sm font-semibold">Join a Family Circle</h2>

      <label className="mt-3 block text-xs text-[var(--color-muted)]">
        Family ID
        <input
          type="number"
          value={familyId}
          onChange={(e) => setFamilyId(e.target.value)}
          className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
        />
      </label>
      <label className="mt-3 block text-xs text-[var(--color-muted)]">
        Passcode
        <input
          type="password"
          value={passcode}
          onChange={(e) => setPasscode(e.target.value)}
          className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
        />
      </label>
      <label className="mt-3 block text-xs text-[var(--color-muted)]">
        Your name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-accent)]"
        />
      </label>

      <div className="mt-4 text-xs text-[var(--color-muted)]">Face (we'll capture 3 photos, 1 second apart)</div>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="mt-2 aspect-video w-full rounded-lg bg-[var(--color-surface-raised)] object-cover"
      />
      {thumbs.length > 0 && (
        <div className="mt-2 flex gap-2">
          {thumbs.map((src, i) => (
            <img key={i} src={src} className="h-16 w-16 rounded-lg object-cover" />
          ))}
        </div>
      )}

      <div className="mt-3 flex gap-2">
        <button
          onClick={startCamera}
          disabled={cameraOn}
          className="rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm font-medium text-[var(--color-text)] hover:bg-[var(--color-surface-raised)] disabled:opacity-50"
        >
          Start Camera
        </button>
        <button
          onClick={capturePhotos}
          disabled={!cameraOn || capturing}
          className="rounded-lg bg-[var(--color-accent)] px-3 py-2 text-sm font-semibold text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
        >
          {capturing ? <Spinner size={14} /> : "Capture 3 Photos"}
        </button>
      </div>

      <button
        onClick={handleSubmit}
        disabled={!ready || submitting}
        className="mt-4 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--color-accent-hover)] disabled:opacity-50"
      >
        {submitting ? <Spinner size={14} /> : "Submit Join Request"}
      </button>

      {error && <div className="mt-3 text-xs text-[var(--color-fake)]">⚠ {error}</div>}
      {notice && <div className="mt-3 text-xs text-[var(--color-real)]">{notice}</div>}
    </Card>
  );
}
