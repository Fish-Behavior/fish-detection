import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  emptyEdits,
  effectiveSegments,
  totals,
  staleFor,
  frameAt,
  frameIndexAt,
  adjacentFrameTime,
  trailRuns,
  videoPoint,
  validateScene,
  validateFrame,
  validateInterval,
  validateReviewer,
} from './model.ts'
import type { Behavior, SessionData } from './model.ts'

test('frame stepping uses stored timestamps, preserves absolute indices and clamps at recording ends', () => {
  const s = session()
  s.frame_count = 360
  s.overlay = { ...s.overlay, frame_idx: [90, 91, 92], t: [3, 3.04, 3.09] }
  assert.equal(adjacentFrameTime(s, 3.04, 1), 3.09)
  assert.equal(adjacentFrameTime(s, 3.04, -1), 3)
  assert.equal(adjacentFrameTime(s, 0, -1), 0)
  assert.equal(adjacentFrameTime(s, s.duration, 1), 359 / s.overlay.fps)
})

// Test-only session: 12 s at 30 fps, fish detected except frames 180–191.
function session(): SessionData {
  const fps = 30,
    n = fps * 12
  const t = Array.from({ length: n }, (_, i) => i / fps)
  const detected = t.map((_, i) => !(i >= 180 && i < 192))
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
    video_id: 'V1',
    name: 'Tank 1',
    video_url: '/v1.mp4',
    duration: 12,
    scene: { roi: [42, 64, 598, 323], waterline: 64 },
    overlay: {
      fps,
      width: 640,
      height: 360,
      t,
      x: t.map((_, i) => (detected[i] ? 100 + i : null)),
      y: t.map((_, i) => (detected[i] ? 180 : null)),
      detected,
      detections: t.flatMap((time, i) =>
        detected[i]
          ? [
              {
                frame_idx: i,
                t: time,
                score: 0.9,
                box: [90 + i, 150, 110 + i, 210] as [
                  number,
                  number,
                  number,
                  number,
                ],
                keypoints: {
                  snout: [100 + i, 180, 0.9] as [number, number, number],
                },
              },
            ]
          : [],
      ),
      detections_error: null,
      has_detector: true,
    },
    segments: cuts.map(([start_s, end_s, state]) => ({
      start_s,
      end_s,
      state,
      source: 'auto' as const,
    })),
    predictions: { video_id: 'V1', predicted: 'A', 'p:A': 0.62, 'p:B': 0.38 },
    measurements: [],
  }
}

test('finite frame/scene/range/reviewer validation rejects invalid corrections', () => {
  const s = session(),
    f = frameAt(s, 0, emptyEdits())
  assert.doesNotThrow(() => validateFrame(f, 640, 360))
  assert.throws(() => validateFrame({ ...f, x: NaN }, 640, 360))
  assert.throws(() => validateFrame({ ...f, detected: false }, 640, 360))
  assert.doesNotThrow(() =>
    validateFrame({ ...f, detected: false, x: null, y: null }, 640, 360),
  )
  assert.throws(() =>
    validateFrame({ ...f, keypoints: { snout: [641, 2, 0.9] } }, 640, 360),
  )
  assert.throws(() =>
    validateScene({ roi: [20, 20, 10, 30], waterline: 64 }, 640, 360),
  )
  assert.throws(() =>
    validateScene({ ...s.scene, waterline: Infinity }, 640, 360),
  )
  for (const [a, b] of [
    [0, 0],
    [3, 1],
    [-1, 2],
    [0, 13],
    [NaN, 1],
  ])
    assert.throws(() => validateInterval(a, b, 12))
  assert.doesNotThrow(() => validateInterval(0, 12, 12))
  assert.throws(() => validateReviewer('  '))
  assert.throws(() => validateReviewer('a\nb'))
})
test('overlapping relabels use latest correction and conserve session duration', () => {
  const s = session(),
    edits = [
      { start_s: 1, end_s: 5, state: 'Dead' as const, reviewer: 'Tester' },
      {
        start_s: 2,
        end_s: 3,
        state: 'Controlled Swim' as const,
        reviewer: 'Tester',
      },
    ]
  const segments = effectiveSegments(s.segments, edits)
  assert.equal(segments.find((s) => s.start_s === 2)?.state, 'Controlled Swim')
  assert.equal(totals(segments).Dead, 3)
  assert.equal(
    Object.values(totals(segments)).reduce((a, b) => a + b),
    12,
  )
  assert.deepEqual(effectiveSegments(s.segments, []), s.segments)
  assert.deepEqual(effectiveSegments([], []), [])
})
test('stale dependencies and manual final result never rewrite original probabilities', () => {
  const s = session(),
    edits = emptyEdits()
  assert.deepEqual(staleFor(edits), {
    measurements: false,
    labels: false,
    predictions: false,
  })
  edits.finalResult = { compound: 'OTHER', dose: '1 mM', reviewer: 'Tester' }
  assert.equal(staleFor(edits).predictions, false)
  assert.equal(s.predictions?.predicted, 'A')
  assert.equal(s.predictions?.['p:A'], 0.62)
  edits.labels.push({ start_s: 0, end_s: 1, state: 'Dead', reviewer: 'Tester' })
  assert.deepEqual(staleFor(edits), {
    measurements: true,
    labels: false,
    predictions: true,
  })
  edits.scene = { ...s.scene, reviewer: 'Tester' }
  assert.equal(staleFor(edits).labels, true)
  assert.equal(staleFor(emptyEdits()).predictions, false)
})
test('missing detections stay null and split trails, including manual missing frames', () => {
  const s = session(),
    e = emptyEdits()
  assert.equal(frameAt(s, 185, e).x, null)
  const runs = trailRuns(s, e, 210)
  assert.equal(runs.length, 2)
  e.frames[205] = {
    ...frameAt(s, 205, e),
    detected: false,
    x: null,
    y: null,
    reviewer: 'Tester',
  }
  assert.equal(trailRuns(s, e, 210).length, 3)
  assert.notEqual(frameAt(s, 205, emptyEdits()).x, null)
})
test('absent detector or tracking data is never replaced by generated geometry', () => {
  const s = session()
  s.overlay.detections = []
  const f = frameAt(s, 10, emptyEdits())
  assert.equal(f.box, null)
  assert.deepEqual(f.keypoints, {})
  assert.doesNotThrow(() => validateFrame(f, 640, 360))
  s.overlay.t = []
  s.overlay.x = []
  s.overlay.y = []
  s.overlay.detected = []
  const none = frameAt(s, 0, emptyEdits())
  assert.deepEqual(
    [none.x, none.y, none.detected, none.box],
    [null, null, false, null],
  )
  assert.deepEqual(trailRuns(s, emptyEdits(), 0), [])
})
test('pointer mapping accounts for letterboxing and responsive scaling', () => {
  const rect = { left: 20, top: 40, width: 800, height: 800 }
  assert.deepEqual(videoPoint([420, 440], rect, 640, 360), [320, 180])
  assert.equal(videoPoint([420, 60], rect, 640, 360), null)
  assert.deepEqual(
    videoPoint(
      [320, 180],
      { left: 0, top: 0, width: 640, height: 360 },
      640,
      360,
    ),
    [320, 180],
  )
})
test('windowed overlays use absolute frame indices and never clamp missing frames', () => {
  const s = session()
  s.overlay.frame_idx = [900, 901]
  s.overlay.t = [30, 30.0333]
  s.overlay.x = [100, 110]; s.overlay.y = [200, 210]; s.overlay.detected = [true, true]
  s.overlay.detections = []
  assert.equal(frameAt(s, 900, emptyEdits()).x, 100)
  assert.equal(frameAt(s, 30, emptyEdits()).x, null)
  assert.equal(frameAt(s, 902, emptyEdits()).detected, false)
  s.frame_count = 1000
  s.overlay.t = [30.02, 30.055]
  assert.equal(frameIndexAt(s, 30.02), 900)
  assert.equal(frameIndexAt(s, 30.055), 901)
})
