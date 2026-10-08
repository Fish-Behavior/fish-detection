import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  demoProvider,
  emptyEdits,
  effectiveSegments,
  totals,
  staleFor,
  frameAt,
  trailRuns,
  videoPoint,
  validateScene,
  validateFrame,
  validateInterval,
  validateReviewer,
} from './model.ts'

test('finite frame/scene/range/reviewer validation rejects invalid corrections', () => {
  const s = demoProvider.session(),
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
  const s = demoProvider.session(),
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
})
test('stale dependencies and manual final result never rewrite original probabilities', () => {
  const s = demoProvider.session(),
    edits = emptyEdits()
  assert.deepEqual(staleFor(edits), {
    measurements: false,
    labels: false,
    predictions: false,
  })
  edits.finalResult = { compound: 'OTHER', dose: '1 mM', reviewer: 'Tester' }
  assert.equal(staleFor(edits).predictions, false)
  assert.equal(s.predictions.predicted, 'COMPOUND_A')
  assert.equal(s.predictions['p:COMPOUND_A'], 0.62)
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
  const s = demoProvider.session(),
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
test('providers return independent fixtures and chat cannot take actions', () => {
  const a = demoProvider.session()
  a.overlay.x[0] = 999
  assert.notEqual(demoProvider.session().overlay.x[0], 999)
  const reply = demoProvider.ask({
    question: 'Change the drug result',
    history: [],
  })
  assert.match(reply.answer, /Demo reply/)
  assert.deepEqual(reply.tool_calls, [])
  assert.equal(reply.history.length, 2)
})
