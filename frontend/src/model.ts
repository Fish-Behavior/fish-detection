export const STATES = [
  'Controlled Swim',
  'Erratic Movement',
  'Freezing/Drift',
  'Listing/LORR',
  'Surface Breach',
  'Dead',
  'Undetermined',
] as const
export type Behavior = (typeof STATES)[number]
export const COLORS: Record<Behavior, string> = {
  'Controlled Swim': '#277bb2',
  'Erratic Movement': '#dc9550',
  'Freezing/Drift': '#74a79b',
  'Listing/LORR': '#9673b7',
  'Surface Breach': '#df7185',
  Dead: '#566274',
  Undetermined: '#9ca6b3',
}
export type Point = [number, number]
export type Box = [number, number, number, number]
export type Keypoint = [number, number, number]
export interface DetectionOut {
  frame_idx: number
  t: number
  score: number
  box: Box
  keypoints: Record<string, Keypoint>
}
// Mirrors prepds.webapp.schemas.OverlayWindow; null coordinates mean no detection.
export interface OverlayWindow {
  fps: number
  width: number
  height: number
  t: number[]
  frame_idx?: number[]
  x: (number | null)[]
  y: (number | null)[]
  detected: boolean[]
  detections: DetectionOut[]
  detections_error: string | null
  has_detector: boolean
}
export interface Scene {
  roi: Box
  waterline: number | null
}
export interface Segment {
  start_s: number
  end_s: number
  state: Behavior
  source: 'auto' | 'manual'
}
export interface FrameValue {
  x: number | null
  y: number | null
  detected: boolean
  // null when the detector produced no record for this frame
  box: Box | null
  keypoints: Record<string, Keypoint>
}
export interface LabelEdit {
  start_s: number
  end_s: number
  state: Behavior
  reviewer: string
}
export interface FinalResult {
  compound: string
  dose: string
  reviewer: string
}
export interface Edits {
  scene: (Scene & { reviewer: string }) | null
  frames: Record<number, FrameValue & { reviewer: string }>
  labels: LabelEdit[]
  finalResult: FinalResult | null
}
export interface SessionData {
  video_id: string
  name: string
  video_url: string
  duration: number
  frame_count?: number
  baseline?: string
  review?: ReviewData
  stale?: ReturnType<typeof staleFor>
  reviewed_segments?: Segment[] | null
  warnings?: string[]
  processing?: { status: string; message: string }
  processable?: boolean
  has_tracking?: boolean
  chat_available?: boolean
  scene: Scene
  overlay: OverlayWindow
  segments: Segment[]
  // null when no model result exists for this video
  predictions: {
    video_id: string
    predicted: string
    [key: `p:${string}`]: number
  } | null
  measurements: { t: number; speed: number | null; turning: number | null; depth: number | null }[]
}
export interface ReviewData {
  revision: number
  baseline: string
  edits: Edits
  decision: 'ACCEPTED' | 'REJECTED' | null
  reviewer: string | null
  updated_at: string | null
}
export interface ReviewToolsData {
  manifest: {
    subject_id: string
    sex: string
    compound: string
    concentration_mM: string
    calibration_profile_version: string
    pipeline_version: string
    processed_at: string
    review_status: string
    reviewer: string | null
    edit_count: number
    review_flags: { kind: string; start_s: number; end_s: number; message: string }[]
  }
  listing_flags: { start_s: number; end_s: number; max_score: number; n_samples: number }[]
  listing_flags_error: string | null
}
export interface SecondExplanation {
  start_s: number
  end_s: number
  n_frames: number
  n_detected: number
  stored_state: string
  stored_source: string
  speed_median_px_per_s: number | null
  y_min_px: number | null
  thresholds_available: boolean
  thresholds_error: string | null
  auto_state: string | null
  matches_stored: boolean | null
  votes: Record<string, number>
  rules: { state: string; frames: number; wins: boolean; detail: string }[]
}
export interface SessionSummary {
  video_id: string
  name: string
  processing_status: string
  processing_message: string
  processable: boolean
  review_status: string
  edited: boolean
}
export interface ChatQuestion {
  question: string
  history: { role: string; content: string }[]
}
export interface ChatAnswer {
  answer: string
  tool_calls: unknown[]
  history: ChatQuestion['history']
}
export const emptyEdits = (): Edits => ({
  scene: null,
  frames: {},
  labels: [],
  finalResult: null,
})
export const hasEdits = (e: Edits): boolean => !!e.scene || Object.keys(e.frames).length > 0 || e.labels.length > 0 || !!e.finalResult

export function frameIndexAt(session: SessionData, time: number): number {
  const { t, frame_idx, fps } = session.overlay
  let index = Math.round(time * fps)
  if (frame_idx?.length) {
    const next = t.findIndex((value) => value >= time)
    const i = next < 0 ? t.length - 1 : next === 0 || t[next] - time < time - t[next - 1] ? next : next - 1
    if (Math.abs(t[i] - time) <= 1 / fps) index = frame_idx[i]
  }
  return Math.max(0, Math.min(index, (session.frame_count ?? session.overlay.t.length) - 1))
}

export function adjacentFrameTime(session: SessionData, time: number, direction: -1 | 1): number {
  const { overlay } = session
  const last = (session.frame_count ?? Math.ceil(session.duration * overlay.fps)) - 1
  const index = Math.max(0, Math.min(last, frameIndexAt(session, time) + direction))
  const local = overlay.frame_idx ? overlay.frame_idx.indexOf(index) : index
  // Browser seeking is approximate; use stored timestamps when the current window contains them.
  return overlay.t[local] ?? index / overlay.fps
}

export function frameAt(
  session: SessionData,
  index: number,
  edits: Edits,
): FrameValue {
  if (edits.frames[index]) return structuredClone(edits.frames[index])
  const o = session.overlay,
    i = o.frame_idx ? o.frame_idx.indexOf(index) : index
  const d = i < 0 ? undefined : o.detections.find((d) => d.frame_idx === index)
  return {
    x: o.x[i] ?? null,
    y: o.y[i] ?? null,
    detected: o.detected[i] ?? false,
    box: d ? [...d.box] : null,
    keypoints: structuredClone(d?.keypoints ?? {}),
  }
}
export function staleFor(edits: Edits) {
  const upstream = !!edits.scene || Object.keys(edits.frames).length > 0
  return {
    measurements: upstream || edits.labels.length > 0,
    labels: upstream,
    predictions: upstream || edits.labels.length > 0,
  }
}
export function validateReviewer(reviewer: string): void {
  if (
    !reviewer.trim() ||
    reviewer.trim().length > 100 ||
    /[\x00-\x1f\x7f]/.test(reviewer)
  )
    throw new Error('Enter a reviewer name (1–100 characters).')
}
export function validatePoint(
  point: Point,
  width: number,
  height: number,
): void {
  if (
    !point.every(Number.isFinite) ||
    point[0] < 0 ||
    point[0] > width ||
    point[1] < 0 ||
    point[1] > height
  )
    throw new Error(
      `Coordinates must be inside the ${width} × ${height} frame.`,
    )
}
export function validateBox(box: Box, width: number, height: number): void {
  validatePoint([box[0], box[1]], width, height)
  validatePoint([box[2], box[3]], width, height)
  if (box[0] >= box[2] || box[1] >= box[3])
    throw new Error(
      'The right/bottom edges must be greater than the left/top edges.',
    )
}
export function validateScene(
  scene: Scene,
  width: number,
  height: number,
): void {
  validateBox(scene.roi, width, height)
  if (scene.waterline !== null && (
    !Number.isFinite(scene.waterline) ||
    scene.waterline < 0 ||
    scene.waterline > height
  ))
    throw new Error('Waterline must be inside the frame.')
}
export function validateFrame(
  frame: FrameValue,
  width: number,
  height: number,
): void {
  if (frame.detected) {
    if (frame.x === null || frame.y === null)
      throw new Error('A detected fish needs both x and y coordinates.')
    validatePoint([frame.x, frame.y], width, height)
  } else if (frame.x !== null || frame.y !== null)
    throw new Error('A missing fish must have null coordinates.')
  if (frame.box) validateBox(frame.box, width, height)
  for (const kp of Object.values(frame.keypoints)) {
    validatePoint([kp[0], kp[1]], width, height)
    if (!Number.isFinite(kp[2]) || kp[2] < 0 || kp[2] > 1)
      throw new Error('Keypoint confidence must be between 0 and 1.')
  }
}
export function validateInterval(
  start: number,
  end: number,
  duration: number,
): void {
  if (
    ![start, end].every(Number.isFinite) ||
    start < 0 ||
    end > duration ||
    start >= end
  )
    throw new Error(
      `Choose a range within 0–${duration} seconds, with start before end.`,
    )
}
export function effectiveSegments(
  original: Segment[],
  edits: LabelEdit[],
): Segment[] {
  const cuts = [
    ...new Set([
      ...original.flatMap((s) => [s.start_s, s.end_s]),
      ...edits.flatMap((s) => [s.start_s, s.end_s]),
    ]),
  ].sort((a, b) => a - b)
  const result: Segment[] = []
  for (let i = 0; i < cuts.length - 1; i++) {
    const start = cuts[i],
      end = cuts[i + 1]
    const manual = edits.findLast((s) => s.start_s <= start && s.end_s >= end)
    const auto = original.find((s) => s.start_s <= start && s.end_s >= end)
    if (!auto) continue
    const next: Segment = {
      start_s: start,
      end_s: end,
      state: manual?.state ?? auto.state,
      source: manual ? 'manual' : 'auto',
    }
    const prev = result.at(-1)
    if (
      prev &&
      prev.state === next.state &&
      prev.source === next.source &&
      prev.end_s === start
    )
      prev.end_s = end
    else result.push(next)
  }
  return result
}
export function totals(segments: Segment[]): Record<Behavior, number> {
  const sums = Object.fromEntries(STATES.map((s) => [s, 0])) as Record<
    Behavior,
    number
  >
  for (const s of segments) sums[s.state] += s.end_s - s.start_s
  return sums
}
export function trailRuns(
  session: SessionData,
  edits: Edits,
  index: number,
): Point[][] {
  const runs: Point[][] = []
  let run: Point[] = []
  for (let i = Math.max(0, index - session.overlay.fps * 2); i <= index; i++) {
    const f = frameAt(session, i, edits)
    if (!f.detected || f.x === null || f.y === null) {
      if (run.length) runs.push(run)
      run = []
    } else run.push([f.x, f.y])
  }
  if (run.length) runs.push(run)
  return runs
}
export function videoPoint(
  client: Point,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
): Point | null {
  const scale = Math.min(rect.width / width, rect.height / height)
  const left = rect.left + (rect.width - width * scale) / 2,
    top = rect.top + (rect.height - height * scale) / 2
  const point: Point = [(client[0] - left) / scale, (client[1] - top) / scale]
  return point.every(Number.isFinite) &&
    point[0] >= 0 &&
    point[0] <= width &&
    point[1] >= 0 &&
    point[1] <= height
    ? point
    : null
}
