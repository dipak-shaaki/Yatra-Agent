import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

type Message = { id: string; role: 'user' | 'assistant'; content: string }
type StreamEvent = { type?: string; content?: string; message?: string }

const SUGGESTIONS = [
  "What's the best time to visit Annapurna Base Camp?",
  'Compare Mardi Himal and Bandipur',
  'How much are Manaslu Circuit permits?',
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sessionToken, setSessionToken] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    bottomRef.current?.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' })
  }, [messages, busy])

  // Reuse the stored session token so a page refresh keeps the
  // server-side conversation history; mint one only on first visit.
  useEffect(() => {
    let cancelled = false
    const stored = localStorage.getItem('yatra_session_token')
    const mint = () =>
      fetch('/api/v1/session', { method: 'POST' })
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
        .then((data: { session_token: string }) => {
          localStorage.setItem('yatra_session_token', data.session_token)
          return data.session_token
        })

    const tokenPromise: Promise<string> = stored ? Promise.resolve(stored) : mint()
    tokenPromise
      .then((token) => {
        if (!cancelled) setSessionToken(token)
      })
      .catch(() => {
        /* chat calls will surface the error */
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function send() {
    const query = input.trim()
    if (!query || busy) return
    setInput('')
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content: query }
    const assistantMessage: Message = { id: crypto.randomUUID(), role: 'assistant', content: '' }
    setMessages((m) => [...m, userMessage, assistantMessage])
    setBusy(true)

    const msgIndex = messages.length + 1

    const append = (delta: string) =>
      setMessages((m) => {
        const next = [...m]
        next[msgIndex] = { ...next[msgIndex], content: next[msgIndex].content + delta }
        return next
      })

    // Yield to the browser's paint loop between events so each chunk
    // visibly renders instead of collapsing into one batched update.
    const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()))

    try {
      if (!sessionToken) throw new Error('Session not ready yet — try again in a moment')
      const res = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Token': sessionToken },
        body: JSON.stringify({ query, stream: true }),
      })
      if (!res.ok || !res.body) {
        if (res.status === 401) localStorage.removeItem('yatra_session_token')
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }

      await readStream(res.body.getReader(), append, nextFrame)
    } catch (err) {
      append(`Error: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">Y</span>
          <span className="brand-text">
            <span className="title">Yatra AI Agent</span>
            <span className="tagline">Nepal domestic travel assistant</span>
          </span>
        </div>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="hint">
            <div className="hint-title">Ask me about Nepal destinations</div>
            <div className="hint-sub">
              Treks, permits, budgets, culture, and best time to visit — try one
              of these:
            </div>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => setInput(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`row ${m.role}`}>
            <div className={`bubble${m.role === 'assistant' ? ' ai' : ''}`}>
              {renderMessageContent(m, busy)}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <input
          value={input}
          placeholder="Message Yatra…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim() || !sessionToken}>
          Send
        </button>
      </footer>
    </div>
  )
}

async function readStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  append: (delta: string) => void,
  nextFrame: () => Promise<void>,
) {
  const decoder = new TextDecoder()
  let buffer = ''
  let done = false
  while (!done) {
    const result = await reader.read()
    buffer += decoder.decode(result.value ?? new Uint8Array(), { stream: true })
    done = result.done
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const raw of events) {
      const shouldStop = await handleStreamEvent(raw, append, nextFrame)
      if (shouldStop) {
        done = true
        break
      }
    }
  }
}

async function handleStreamEvent(
  raw: string,
  append: (delta: string) => void,
  nextFrame: () => Promise<void>,
): Promise<boolean> {
  const dataLine = raw.split('\n').find((line) => line.startsWith('data: '))
  if (!dataLine) return false
  try {
    const event = JSON.parse(dataLine.slice(6)) as StreamEvent
    if (event.type === 'token' && event.content) {
      append(event.content)
      await nextFrame()
    } else if (event.type === 'error') {
      append(`\n\nError: ${event.message}`)
    } else if (event.type === 'done') {
      return true
    }
  } catch {
    // Ignore malformed events, matching the existing streaming behavior.
  }
  return false
}

function renderMessageContent(message: Message, busy: boolean) {
  if (message.role !== 'assistant') return message.content
  if (message.content) return <ReactMarkdown>{message.content}</ReactMarkdown>
  if (!busy) return ''
  return (
    <div className="typing">
      <span />
      <span />
      <span />
    </div>
  )
}

export default App
