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
  x: (number | null)[]
  y: (number | null)[]
  detected: boolean[]
  detections: DetectionOut[]
  detections_error: string | null
  has_detector: boolean
}
export interface Scene {
  roi: Box
  waterline: number
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
  box: Box
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
export interface DemoSession {
  video_id: string
  name: string
  video_url: string
  duration: number
  scene: Scene
  overlay: OverlayWindow
  segments: Segment[]
  predictions: {
    video_id: string
    predicted: string
    [key: `p:${string}`]: number
  }
  measurements: { t: number; speed: number; turning: number; depth: number }[]
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

export function position(t: number): Point {
  return [320 + 170 * Math.sin(t * 0.5), 180 + 65 * Math.sin(t * 0.9)]
}
export function sampleFrame(t: number): FrameValue {
  const [x, y] = position(t)
  const angle = Math.atan2(58.5 * Math.cos(t * 0.9), 85 * Math.cos(t * 0.5))
  const point = (a: number, b: number): Keypoint => [
    x + a * Math.cos(angle) - b * Math.sin(angle),
    y + a * Math.sin(angle) + b * Math.cos(angle),
    0.92,
  ]
  const detected = !(t >= 6 && t < 6.4)
  return {
    x: detected ? x : null,
    y: detected ? y : null,
    detected,
    box: [x - 35, y - 30, x + 35, y + 30],
    keypoints: {
      snout: point(20, 0),
      tail_base: point(-17, 0),
      tail_tip: point(-30, 0),
      dorsal_fin_base: point(0, -7),
      ventral: point(0, 7),
    },
  }
}

function makeDemo(): DemoSession {
  const fps = 30,
    duration = 12
  const overlay: OverlayWindow = {
    fps,
    width: 640,
    height: 360,
    t: [],
    x: [],
    y: [],
    detected: [],
    detections: [],
    detections_error: null,
    has_detector: true,
  }
  for (let i = 0; i < fps * duration; i++) {
    const t = i / fps,
      frame = sampleFrame(t)
    overlay.t.push(t)
    overlay.x.push(frame.x)
    overlay.y.push(frame.y)
    overlay.detected.push(frame.detected)
    if (frame.detected)
      overlay.detections.push({
        frame_idx: i,
        t,
        score: 0.94,
        box: frame.box,
        keypoints: frame.keypoints,
      })
  }
  const cuts: [number, number, Behavior][] = [
    [0, 2.5, 'Controlled Swim'],
    [2.5, 4, 'Erratic Movement'],
    [4, 6, 'Freezing/Drift'],
    [6, 6.4, 'Undetermined'],
    [6.4, 8, 'Listing/LORR'],
    [8, 9.5, 'Surface Breach'],
    [9.5, 12, 'Controlled Swim'],
  ]
  return {
    video_id: 'SYNTH_001',
    name: 'Synthetic tank · 001',
    video_url: '/demo-tank.mp4',
    duration,
    scene: { roi: [42, 64, 598, 323], waterline: 64 },
    overlay,
    segments: cuts.map(([start_s, end_s, state]) => ({
      start_s,
      end_s,
      state,
      source: 'auto',
    })),
    predictions: {
      video_id: 'SYNTH_001',
      predicted: 'COMPOUND_A',
      'p:COMPOUND_A': 0.62,
      'p:COMPOUND_B': 0.24,
      'p:VEHICLE': 0.14,
    },
    measurements: Array.from({ length: 49 }, (_, i) => {
      const t = i / 4
      return {
        t,
        speed: Math.hypot(85 * Math.cos(t * 0.5), 58.5 * Math.cos(t * 0.9)),
        turning: 18 + 12 * Math.sin(t * 0.8),
        depth: position(t)[1] - 64,
      }
    }),
  }
}
const fixture = makeDemo()
// ponytail: fixed fixtures only; replace this provider with real APIs after scaffold review.
export const demoProvider = {
  session: (): DemoSession => structuredClone(fixture),
  ask: ({ question, history }: ChatQuestion): ChatAnswer => {
    const answer = `Demo reply · The research chat backend is not connected. This sample has a 12-second synthetic clip and illustrative compound probabilities. Corrections stay in this tab; I cannot run inference or edit results. Your question: “${question}”`
    return {
      answer,
      tool_calls: [],
      history: [
        ...history,
        { role: 'user', content: question },
        { role: 'assistant', content: answer },
      ],
    }
  },
}

export function frameAt(
  session: DemoSession,
  index: number,
  edits: Edits,
): FrameValue {
  if (edits.frames[index]) return structuredClone(edits.frames[index])
  const o = session.overlay,
    i = Math.max(0, Math.min(index, o.t.length - 1))
  const d = o.detections.find((d) => d.frame_idx === i)
  const fallback = sampleFrame(o.t[i])
  return {
    x: o.x[i],
    y: o.y[i],
    detected: o.detected[i],
    box: [...(d?.box ?? fallback.box)],
    keypoints: structuredClone(d?.keypoints ?? fallback.keypoints),
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
  if (
    !Number.isFinite(scene.waterline) ||
    scene.waterline < 0 ||
    scene.waterline > height
  )
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
  validateBox(frame.box, width, height)
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
  session: DemoSession,
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
