import {
  ApiError,
  type AnalyzeImageResult,
  type AnalyzeVideoResult,
  type Contact,
  type Family,
  type FamilyMember,
  type HistoryEntry,
  type ModelInfo,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON -- keep the generic HTTP status message
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/** POST with upload progress via XHR -- fetch's request-body streaming
 * progress isn't reliably available across browsers the way
 * xhr.upload.onprogress is. */
function uploadWithProgress<T>(
  path: string,
  formData: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}${path}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = { detail: xhr.responseText || `HTTP ${xhr.status}` };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as T);
      } else {
        const detail =
          typeof body === "object" && body !== null && "detail" in body
            ? String((body as { detail: unknown }).detail)
            : `HTTP ${xhr.status}`;
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error -- is the backend running?"));
    xhr.send(formData);
  });
}

export function health(): Promise<{ status: string }> {
  return request("/health");
}

export function modelInfo(): Promise<ModelInfo> {
  return request("/model-info");
}

export function analyzeImage(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<AnalyzeImageResult> {
  const formData = new FormData();
  formData.append("file", file);
  return uploadWithProgress("/analyze-image", formData, onProgress);
}

export function analyzeVideo(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<AnalyzeVideoResult> {
  const formData = new FormData();
  formData.append("file", file);
  return uploadWithProgress("/analyze-video", formData, onProgress);
}

export function getHistory(limit = 50): Promise<HistoryEntry[]> {
  return request(`/history?limit=${limit}`);
}

export function listContacts(): Promise<Contact[]> {
  return request("/contacts");
}

export function enrollContact(name: string, file: File): Promise<{ id: number; name: string }> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", file);
  return request("/contacts", { method: "POST", body: formData });
}

export function createFamily(name: string, passcode: string): Promise<Family> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("passcode", passcode);
  return request("/families", { method: "POST", body: formData });
}

export function joinFamily(
  familyId: number,
  name: string,
  passcode: string,
  photos: Blob[],
): Promise<{ id: number; name: string; status: string }> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("passcode", passcode);
  photos.forEach((blob, i) => formData.append("photos", blob, `photo${i}.jpg`));
  return request(`/families/${familyId}/join`, { method: "POST", body: formData });
}

export function getFamilyMembers(familyId: number, passcode: string): Promise<FamilyMember[]> {
  return request(`/families/${familyId}/members?passcode=${encodeURIComponent(passcode)}`);
}

export function approveMember(familyId: number, memberId: number, passcode: string) {
  const formData = new FormData();
  formData.append("passcode", passcode);
  return request(`/families/${familyId}/members/${memberId}/approve`, {
    method: "POST",
    body: formData,
  });
}

export function rejectMember(familyId: number, memberId: number, passcode: string) {
  const formData = new FormData();
  formData.append("passcode", passcode);
  return request(`/families/${familyId}/members/${memberId}/reject`, {
    method: "POST",
    body: formData,
  });
}
