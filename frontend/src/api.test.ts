import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ApiError, askChat, explainSecond, loadReviewTools, loadSession, request } from './api.ts'

test('empty inventory means no session; session load uses the selected ID', async (t) => {
  const calls: string[] = []
  t.mock.method(globalThis, 'fetch', async (path: string) => {
    calls.push(path)
    return new Response(JSON.stringify(path === '/api/sessions' ? [] : { video_id: 'F_1' }), { status: 200 })
  })
  assert.equal(await loadSession(), null)
  assert.equal((await loadSession('F_1'))?.video_id, 'F_1')
  assert.deepEqual(calls, ['/api/sessions', '/api/sessions/F_1'])
})

test('HTTP conflicts, validation errors, invalid responses and offline errors stay actionable', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ detail: { code: 'revision_conflict', message: 'Reload first.' } }), { status: 409 }))
  await assert.rejects(request('/api/test'), (e: unknown) => e instanceof ApiError && e.status === 409 && e.code === 'revision_conflict' && e.message === 'Reload first.')
  fetch.mock.mockImplementation(async () => new Response(JSON.stringify({ detail: [{ loc: ['body', 'reviewer'], msg: 'blank' }] }), { status: 422 }))
  await assert.rejects(request('/api/test'), /body.reviewer: blank/)
  fetch.mock.mockImplementation(async () => new Response('<html>error</html>', { status: 502 }))
  await assert.rejects(request('/api/test'), /invalid response/)
  fetch.mock.mockImplementation(async () => { throw new TypeError('Failed to fetch') })
  await assert.rejects(request('/api/test'), /Cannot reach the backend/)
  fetch.mock.mockImplementation(async () => { throw new DOMException('timeout', 'TimeoutError') })
  await assert.rejects(request('/api/test'), /Reload to check whether the backend saved/)
})

test('chat preserves the DCS question and history contract', async (t) => {
  let body = ''
  t.mock.method(globalThis, 'fetch', async (_path: string, init: RequestInit) => {
    body = init.body as string
    return new Response(JSON.stringify({ answer: 'Answer', tool_calls: [], history: [{ role: 'assistant', content: 'Answer' }] }))
  })
  assert.equal((await askChat({ question: 'Question', history: [] })).answer, 'Answer')
  assert.deepEqual(JSON.parse(body), { question: 'Question', history: [] })
})

test('prepds review readers use the shared API namespace and encoded recording ID', async (t) => {
  const calls: string[] = []
  t.mock.method(globalThis, 'fetch', async (path: string) => {
    calls.push(path)
    return new Response('{}')
  })
  await loadReviewTools('Fish 1')
  await explainSecond('Fish 1', 4)
  await loadReviewTools('Fish 1', 'saved baseline')
  await explainSecond('Fish 1', 4, 'saved baseline')
  assert.deepEqual(calls, ['/api/sessions/Fish%201/review-tools', '/api/sessions/Fish%201/explain?second=4',
    '/api/sessions/Fish%201/review-tools?baseline=saved%20baseline', '/api/sessions/Fish%201/explain?second=4&baseline=saved%20baseline'])
})
