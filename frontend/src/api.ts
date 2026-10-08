import type { ChatAnswer, ChatQuestion, Edits, OverlayWindow, ReviewData, ReviewToolsData, SecondExplanation, SessionData, SessionSummary } from './model.ts'

export class ApiError extends Error {
  status: number
  code: string
  constructor(message: string, status = 0, code = 'request_failed') { super(message); this.status = status; this.code = code }
}
export async function request<T>(path: string, init: RequestInit = {}, timeout = 30000): Promise<T> {
  try {
    const response = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...init.headers }, signal: AbortSignal.timeout(timeout) })
    let body
    try { body = await response.json() } catch { throw new ApiError('The backend returned an invalid response. Check the API connection.', response.status) }
    if (!response.ok) {
      const detail = body?.detail
      const message = typeof detail === 'string' ? detail : Array.isArray(detail)
        ? detail.map((e: { loc: string[]; msg: string }) => `${e.loc.join('.')}: ${e.msg}`).join('; ')
        : detail?.message ?? `The backend request failed (${response.status}).`
      throw new ApiError(message, response.status, detail?.code)
    }
    return body as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof Error && ['TimeoutError', 'AbortError'].includes(error.name))
      throw new ApiError('The request timed out. Reload to check whether the backend saved it before retrying.', 0, 'timeout')
    throw new ApiError('Cannot reach the backend. Run npm start in frontend/ and reload.', 0, 'offline')
  }
}
const path = (id: string) => `/api/sessions/${encodeURIComponent(id)}`
export const listSessions = () => request<SessionSummary[]>('/api/sessions')
export async function loadSession(id?: string): Promise<SessionData | null> {
  id ??= (await listSessions())[0]?.video_id
  return id ? request<SessionData>(path(id)) : null
}
export const loadOverlay = (id: string, start: number, end: number) => request<OverlayWindow>(`${path(id)}/overlay?start_s=${start}&end_s=${end}`)
export const loadReviewTools = (id: string, baseline?: string) => request<ReviewToolsData>(`${path(id)}/review-tools${baseline ? `?baseline=${encodeURIComponent(baseline)}` : ''}`)
export const explainSecond = (id: string, second: number, baseline?: string) => request<SecondExplanation>(`${path(id)}/explain?second=${second}${baseline ? `&baseline=${encodeURIComponent(baseline)}` : ''}`)
export const saveReview = (session: SessionData, review: ReviewData, edits: Edits, reviewer: string, decision: ReviewData['decision']) =>
  request<ReviewData>(`${path(session.video_id)}/review`, { method: 'PUT', body: JSON.stringify({ revision: review.revision, baseline: session.baseline, reviewer, edits, decision }) })
export const rerun = (session: SessionData, review: ReviewData, reviewer: string) =>
  request<SessionData>(`${path(session.video_id)}/rerun`, { method: 'POST', body: JSON.stringify({ revision: review.revision, baseline: session.baseline, reviewer }) }, 120000)
export const predict = (session: SessionData, review: ReviewData, reviewer: string) =>
  request<SessionData>(`${path(session.video_id)}/predict`, { method: 'POST', body: JSON.stringify({ revision: review.revision, baseline: session.baseline, reviewer }) }, 260000)
export const startProcessing = (id: string, reviewer: string) => request<{ status: string; message: string }>(`${path(id)}/process`, { method: 'POST', body: JSON.stringify({ reviewer }) })
export const processingStatus = (id: string) => request<{ status: string; message: string }>(`${path(id)}/processing`)
export const askChat = (q: ChatQuestion) => request<ChatAnswer>('/api/ask', { method: 'POST', body: JSON.stringify(q) }, 100000)
