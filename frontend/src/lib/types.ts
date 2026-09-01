// Mirrors backend/main.py's documented endpoint contract (kept in sync by
// hand -- see that file's module docstring for the source of truth).

export type Classification =
  | "real"
  | "fake"
  | "uncertain"
  | "insufficient_quality"
  | "no_face_detected";

export interface NormalizedBbox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AnalyzedFace {
  bbox: NormalizedBbox;
  detection_confidence: number;
  fake_probability: number | null;
  classification: Classification;
  quality_issue: string | null;
}

export interface AnalyzeImageResult {
  faces: AnalyzedFace[];
  overall_classification: Classification;
  overall_confidence: number | null;
  processing_time_ms: number;
  model_version: string | null;
  request_id: string;
}

export interface VideoFrameResult {
  timestamp: number;
  fake_probability: number | null;
  classification: Classification;
  bbox: NormalizedBbox | null;
}

/** A single sampled frame kept as evidence: a downsized JPEG of the whole
 * frame (not just the face crop) plus the detected face's bbox, so the UI
 * can show *which* frame and *where in it* the model reacted to. */
export interface FrameEvidence {
  timestamp: number;
  fake_probability: number | null;
  bbox: NormalizedBbox | null;
  /** base64-encoded JPEG, no data: prefix. */
  thumbnail: string;
}

export interface SuspiciousSegment {
  start_timestamp: number;
  end_timestamp: number;
  peak_fake_probability: number;
  frame_count: number;
  peak_frame: FrameEvidence | null;
}

export interface VideoMetadata {
  duration_seconds: number;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  frames_analyzed: number;
}

export interface AnalyzeVideoResult {
  metadata: VideoMetadata;
  frames: VideoFrameResult[];
  suspicious_segments: SuspiciousSegment[];
  most_suspicious_frame: FrameEvidence | null;
  overall_classification: Classification;
  overall_confidence: number | null;
  processing_time_ms: number;
  model_version: string | null;
  request_id: string;
}

export interface Contact {
  id: number;
  name: string;
  created_at: string;
}

export interface HistoryEntry {
  id: number;
  created_at: string;
  platform: string | null;
  fake_probability: number;
  thumbnail_b64: string;
  identity_contact_name: string | null;
  identity_similarity: number | null;
  scam_likelihood: number | null;
}

export interface Family {
  id: number;
  name: string;
}

export type MemberStatus = "pending" | "approved" | "rejected";

export interface FamilyMember {
  id: number;
  name: string;
  status: MemberStatus;
  created_at: string;
  approved_at: string | null;
  has_face: 0 | 1;
}

export interface ModelInfo {
  model_version: string;
  backbone: string;
  checkpoint: string;
  input_size: number;
  device: string;
  cuda_available: boolean;
  status: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}
