import { useEffect, useState } from "react";
import { VOICE_API_BASE } from "./voiceCheck";

export type VoiceStatus = "checking" | "online" | "offline";

/** Mirrors useBackendStatus.ts, pointed at the voice anti-spoofing service
 * (voice-integrity/src/server.py) instead. That service's own lifespan
 * loads both Gustking and XLS-R+SLS before it starts responding to
 * requests at all (see server.py) -- so a successful /health here already
 * means both voice models are loaded, not just that the process is up. */
export function useVoiceStatus(pollMs = 15000): VoiceStatus {
  const [status, setStatus] = useState<VoiceStatus>("checking");

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const res = await fetch(`${VOICE_API_BASE}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        if (!cancelled) setStatus("online");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    }

    check();
    const id = setInterval(check, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollMs]);

  return status;
}
