import type { ChatAnswer, ChatQuestion, SessionData } from './model.ts'

// Backend seam: the UI renders nothing until these return data (see BACKEND_HANDOFF.md).

// Resolve null when no session exists; reject on failure (the UI shows the error).
export const loadSession = async (): Promise<SessionData | null> => null

// Leave null to hide the chat bubble until the DCS /api/ask adapter exists.
export const askChat: ((q: ChatQuestion) => Promise<ChatAnswer>) | null = null
