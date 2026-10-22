import { useEffect, useRef, useState } from 'react'
import type { ChatAnswer, ChatQuestion } from './model.ts'
import ChatMarkdown from './ChatMarkdown.tsx'
import { Icon } from './ui.tsx'

type ToolCall = { tool?: string; arguments?: unknown; error?: string | null }

// What each query tool does, in the researcher's words, for the "How this was answered" list.
const TOOL_LABELS: Record<string, string> = {
  list_compounds: 'Listed compounds and fish counts',
  find_features: 'Looked up the measure name',
  compare_to_vehicle: 'Compared a compound with vehicle',
  top_differences: 'Ranked the largest changes vs vehicle',
  feature_by_compound: 'Compared every compound on one measure',
  fish_profile: 'Read one fish\'s profile',
  fish_timeline: 'Read one fish\'s behavior over time',
  model_results: 'Read the classifier results',
  ablation_results: 'Read the ablation results',
}

const SUGGESTIONS = [
  'Which compounds are in the dataset, and how many fish each?',
  'Which compound changed swimming speed the most?',
  'Can the model tell compounds apart better than guessing the recording date?',
]

const describe = (call: ToolCall): string => {
  const args = call.arguments && typeof call.arguments === 'object'
    ? Object.entries(call.arguments as Record<string, unknown>).map(([k, v]) => `${k}: ${String(v)}`).join(', ')
    : typeof call.arguments === 'string' ? call.arguments : ''
  return `${TOOL_LABELS[call.tool ?? ''] ?? call.tool ?? 'Tool'}${args ? ` (${args})` : ''}`
}

export default function Chat({
  ask,
  available,
  open,
  onOpenChange,
}: {
  ask: (q: ChatQuestion) => Promise<ChatAnswer>
  available: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState<ChatQuestion['history']>([])
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('')
  const [pending, setPending] = useState(''),
    [elapsed, setElapsed] = useState(0)
  // Tool calls behind each assistant answer, keyed by its index in `history` (the API returns them per answer).
  const [sources, setSources] = useState<Record<number, ToolCall[]>>({})
  const bubble = useRef<HTMLButtonElement>(null),
    input = useRef<HTMLInputElement>(null),
    conversation = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])
  useEffect(() => {
    if (conversation.current)
      conversation.current.scrollTop = conversation.current.scrollHeight
  }, [history, pending, busy])
  useEffect(() => {
    if (!busy) return
    setElapsed(0)
    const started = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [busy])
  const send = async (text: string) => {
    const q = text.trim()
    if (!available || !q || busy) return
    setBusy(true)
    setError('')
    setPending(q)
    setQuestion('')
    try {
      const answer = await ask({ question: q, history })
      setHistory(answer.history)
      setSources((s) => ({ ...s, [answer.history.length - 1]: answer.tool_calls as ToolCall[] }))
    } catch (err) {
      setQuestion(q)
      setError(
        err instanceof Error
          ? err.message
          : 'The assistant is unavailable.',
      )
    } finally {
      setPending('')
      setBusy(false)
    }
  }
  const close = () => {
    onOpenChange(false)
    bubble.current?.focus()
  }
  return (
    <>
      {open && (
        <section
          className="chat-panel"
          role="dialog"
          aria-label="Research chat"
          onKeyDown={(e) => {
            if (e.key === 'Escape') close()
          }}
        >
          <div className="chat-header">
            <span className="chat-symbol">
              <Icon name="chat" />
            </span>
            <div>
              <h2>Research assistant</h2>
              <span>{available ? 'Answers only' : 'Not connected'}</span>
            </div>
            <button
              className="icon-button"
              aria-label="Close research chat"
              onClick={close}
            >
              <Icon name="close" />
            </button>
          </div>
          <div
            className="chat-conversation"
            ref={conversation}
            role="log"
            aria-live="polite"
          >
            <div className="chat-welcome">
              <h3>A little help, wherever you need it.</h3>
              <p>{available
                ? 'Ask about stored behavior data, compounds, or model results.'
                : 'The research chat model is not configured yet. Connect the chat service to enable answers.'}</p>
              {available && (
                <div className="chat-suggestions">
                  {SUGGESTIONS.map((text) => (
                    <button key={text} disabled={busy} onClick={() => void send(text)}>
                      {text}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {history.map((message, i) => (
              <div key={i} className={`chat-message ${message.role}`}>
                <b>{message.role === 'user' ? 'You' : 'Assistant'}</b>
                {message.role === 'user' ? <p>{message.content}</p> : <ChatMarkdown text={message.content} />}
                {message.role !== 'user' && sources[i] && (
                  <details className="chat-sources">
                    <summary>
                      {sources[i].length ? `How this was answered (${sources[i].length} data ${sources[i].length === 1 ? 'lookup' : 'lookups'})` : 'Answered without looking up data'}
                    </summary>
                    <ul>
                      {sources[i].map((call, j) => (
                        <li key={j} className={call.error ? 'failed' : undefined}>
                          {describe(call)}
                          {call.error ? ` - failed: ${call.error}` : ''}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            ))}
            {pending && (
              <div className="chat-message user">
                <b>You</b>
                <p>{pending}</p>
              </div>
            )}
            {busy && (
              <div className="chat-message assistant chat-thinking" role="status">
                <b>Assistant</b>
                <p>
                  <span className="chat-dots" aria-hidden="true"><i /><i /><i /></span>
                  Querying the study data and writing the answer… {elapsed}s
                </p>
              </div>
            )}
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
          </div>
          <form
            className="chat-form"
            onSubmit={(e) => {
              e.preventDefault()
              void send(question)
            }}
          >
            <label className="sr-only" htmlFor="chat-question">
              Research question
            </label>
            <input
              ref={input}
              id="chat-question"
              value={question}
              disabled={!available}
              onChange={(e) => setQuestion(e.target.value)}
              maxLength={2000}
              placeholder="Ask a research question…"
            />
            <button
              type="submit"
              aria-label="Send question"
              disabled={!available || !question.trim() || busy}
            >
              <Icon name="arrow" />
            </button>
          </form>
          <div className="chat-footer">
            <button
              className="text-button"
              disabled={busy}
              onClick={() => {
                setHistory([])
                setSources({})
              }}
            >
              Clear chat
            </button>
            <span>Refresh clears this chat.</span>
          </div>
        </section>
      )}
      <button
        ref={bubble}
        className={`chat-bubble ${open ? 'open' : ''}`}
        aria-label={open ? 'Close research chat' : 'Open research chat'}
        aria-expanded={open}
        onClick={() => {
          if (open) close()
          else onOpenChange(true)
        }}
      >
        <Icon name={open ? 'close' : 'chat'} size={24} />
      </button>
    </>
  )
}
