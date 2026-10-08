import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import { emptyEdits, frameAt } from './model.ts'
import type { SessionData } from './model.ts'

test('Review exposes advanced controls and staged labels; Dashboard hides unsaved drafts', async () => {
  const server = await createServer({ configFile: false, optimizeDeps: { noDiscovery: true }, server: { middlewareMode: true, hmr: false, ws: false, watch: null } })
  try {
    const { default: VideoPanel } = await server.ssrLoadModule('/src/VideoPanel.tsx')
    const { LabelEditor } = await server.ssrLoadModule('/src/Editors.tsx')
    const session: SessionData = {
      video_id: 'F1', name: 'Test fish', video_url: '/test.mp4', duration: 2, frame_count: 2,
      scene: { waterline: 10, roi: [0, 0, 100, 100] },
      overlay: { fps: 1, width: 100, height: 100, t: [0, 1], frame_idx: [0, 1],
        x: [20, 30], y: [20, 30], detected: [true, true], detections: [], detections_error: null, has_detector: false },
      segments: [{ start_s: 0, end_s: 2, state: 'Controlled Swim', source: 'auto' }],
      predictions: null, measurements: [],
    }
    const edits = emptyEdits()
    const noop = () => {}
    const props = { session, edits, src: session.video_url, title: session.name, time: 0, duration: 2,
      setTime: noop, setDuration: noop, videoRef: { current: null }, seek: noop, tool: 'point', keypoint: 'snout',
      sceneDraft: { ...session.scene, waterline: null }, frameDraft: { ...frameAt(session, 0, edits), detected: false },
      sketch: noop, nudge: noop, markRange: noop }
    const dashboard = renderToStaticMarkup(createElement(VideoPanel, { ...props, reviewMode: false }))
    const review = renderToStaticMarkup(createElement(VideoPanel, { ...props, reviewMode: true }))
    assert.match(dashboard, /Fish tracker/)
    assert.doesNotMatch(dashboard, /Previous frame|Mark range start|overlay-toggles|Draft:|video-overlay editing/)
    assert.match(review, /Tracking review workspace/)
    assert.match(review, /Previous frame/)
    assert.match(review, /Mark range start/)
    assert.match(review, /Draft: fish marked missing/)
    assert.match(review, /Draft: waterline cleared/)
    const pending = [{ start_s: 0, end_s: 1, state: 'Listing/LORR', reviewer: 'R' }]
    const editor = renderToStaticMarkup(createElement(LabelEditor, {
      session, edits, range: [0, 1], setRange: noop, state: 'Listing/LORR', setState: noop,
      saveLabels: noop, resetLabels: noop, pending, stage: noop, saveStaged: noop, removeStaged: noop, discardStaged: noop,
    }))
    assert.match(editor, /Staged edits · not saved/)
    assert.match(editor, /Save staged edits/)
    assert.match(editor, /Discard staged edits/)
    assert.match(editor, /aria-label="Remove staged edit 1"/)
    assert.match(editor, /disabled="">Apply behavior/)
  } finally { await server.close() }
})

test('Research chat stays visible without a model and enables questions when connected', async () => {
  const server = await createServer({ configFile: false, optimizeDeps: { noDiscovery: true }, server: { middlewareMode: true, hmr: false, ws: false, watch: null } })
  try {
    const { default: Chat } = await server.ssrLoadModule('/src/Chat.tsx')
    const props = { ask: async () => ({ answer: '', tool_calls: [], history: [] }), onOpenChange: () => {} }
    const closed = renderToStaticMarkup(createElement(Chat, { ...props, open: false, available: false }))
    assert.match(closed, /aria-label="Open research chat"/)
    const offline = renderToStaticMarkup(createElement(Chat, { ...props, open: true, available: false }))
    assert.match(offline, /Not connected/)
    assert.match(offline, /model is not configured yet/)
    assert.match(offline.match(/<input[^>]*id="chat-question"[^>]*>/)![0], /disabled=""/)
    const connected = renderToStaticMarkup(createElement(Chat, { ...props, open: true, available: true }))
    assert.match(connected, /Ask about stored behavior data/)
    assert.doesNotMatch(connected.match(/<input[^>]*id="chat-question"[^>]*>/)![0], /disabled=/)
  } finally { await server.close() }
})
