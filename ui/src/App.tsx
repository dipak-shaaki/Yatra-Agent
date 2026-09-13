import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

type Message = { role: 'user' | 'assistant'; content: string }

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sender, setSender] = useState('test_user')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function send() {
    const query = input.trim()
    if (!query || busy) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', content: query }])
    setBusy(true)
    try {
      const res = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, sender }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
      setMessages((m) => [...m, { role: 'assistant', content: data.answer }])
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Error: ${(err as Error).message}` },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="title">Yatra</span>
        <label className="sender">
          sender
          <input
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            spellCheck={false}
          />
        </label>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <p className="hint">
            Ask about the 10 destinations — e.g. "What's the best time for
            Annapurna Base Camp?"
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`row ${m.role}`}>
            <div className="bubble">
              {m.role === 'assistant' ? (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="row assistant">
            <div className="bubble typing">…</div>
          </div>
        )}
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
        <button onClick={send} disabled={busy || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  )
}

export default App