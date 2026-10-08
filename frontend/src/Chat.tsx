import { useEffect, useRef, useState } from 'react'
import type { ChatAnswer, ChatQuestion } from './model.ts'
import { Icon } from './ui.tsx'

export default function Chat({
  ask,
  open,
  onOpenChange,
}: {
  ask: (q: ChatQuestion) => Promise<ChatAnswer>
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState<ChatQuestion['history']>([])
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('')
  const bubble = useRef<HTMLButtonElement>(null),
    input = useRef<HTMLInputElement>(null),
    conversation = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])
  useEffect(() => {
    if (conversation.current)
      conversation.current.scrollTop = conversation.current.scrollHeight
  }, [history])
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
              <span>Answers only</span>
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
              <p>Ask about this session or the review workflow.</p>
              <button
                onClick={() => {
                  setQuestion('What happens when I correct a tracking point?')
                  input.current?.focus()
                }}
              >
                What happens after a correction?
              </button>
            </div>
            {history.map((message, i) => (
              <div key={i} className={`chat-message ${message.role}`}>
                <b>{message.role === 'user' ? 'You' : 'Assistant'}</b>
                <p>{message.content}</p>
              </div>
            ))}
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
          </div>
          <form
            className="chat-form"
            onSubmit={async (e) => {
              e.preventDefault()
              const q = question.trim()
              if (!q || busy) return
              setBusy(true)
              setError('')
              try {
                setHistory((await ask({ question: q, history })).history)
                setQuestion('')
              } catch (err) {
                setError(
                  err instanceof Error
                    ? err.message
                    : 'The assistant is unavailable.',
                )
              } finally {
                setBusy(false)
              }
            }}
          >
            <label className="sr-only" htmlFor="chat-question">
              Research question
            </label>
            <input
              ref={input}
              id="chat-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              maxLength={2000}
              placeholder="Ask a research question…"
            />
            <button
              type="submit"
              aria-label="Send question"
              disabled={!question.trim() || busy}
            >
              <Icon name="arrow" />
            </button>
          </form>
          <div className="chat-footer">
            <button className="text-button" onClick={() => setHistory([])}>
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
