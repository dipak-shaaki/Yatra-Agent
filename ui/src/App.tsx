import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

type Message = { role: 'user' | 'assistant'; content: string }

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
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  // Mint a session token once per page load; the server derives the chat
  // identity from it, so there is no client-chosen sender anymore.
  useEffect(() => {
    let cancelled = false
    fetch('/api/v1/session', { method: 'POST' })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data: { session_token: string }) => {
        if (!cancelled) setSessionToken(data.session_token)
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
    setMessages((m) => [...m, { role: 'user', content: query }])
    setMessages((m) => [...m, { role: 'assistant', content: '' }])
    setBusy(true)

    const msgIndex = messages.length + 1

    const append = (delta: string) =>
      setMessages((m) => {
        const next = [...m]
        next[msgIndex] = { role: 'assistant', content: next[msgIndex].content + delta }
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
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let done = false

      while (!done) {
        const { done: readDone, value } = await reader.read()
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: true })
        done = readDone

        const events = buffer.split('\n\n')
        buffer = events.pop() ?? ''

        for (const raw of events) {
          const line = raw.split('\n').find((l) => l.startsWith('data: '))
          if (!line) continue
          const payload = line.slice(6)
          if (payload === '[DONE]') {
            done = true
            break
          }
          try {
            const { chunk } = JSON.parse(payload)
            if (chunk) {
              append(chunk)
              await nextFrame()
            }
          } catch {
            /* ignore malformed events */
          }
        }
      }
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
        {messages.map((m, i) => (
          <div key={i} className={`row ${m.role}`}>
            <div className={`bubble${m.role === 'assistant' ? ' ai' : ''}`}>
              {m.role === 'assistant' ? (
                m.content ? (
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                ) : busy ? (
                  <div className="typing">
                    <span />
                    <span />
                    <span />
                  </div>
                ) : (
                  ''
                )
              ) : (
                m.content
              )}
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

export default App