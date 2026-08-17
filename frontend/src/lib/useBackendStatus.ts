import { useEffect, useState } from "react";
import { health } from "./api";

export type BackendStatus = "checking" | "online" | "offline";

export function useBackendStatus(pollMs = 15000): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        await health();
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
