import { useState, useEffect, useRef } from 'react'

// ─── Settings helpers ────────────────────────────────────────────
const SETTINGS_VER = 'v7'
const DEFAULTS = {
  re_rank:     true,
  k:           5,
  pool_size:   15,
  data_dir:    'data/llm_ai_2023_2026',
  gemini_key:  '',
  cohere_key:  '',
  answer_mode: 'gemini',
}

function loadSettings() {
  const saved   = localStorage.getItem('rag_settings')
  const version = localStorage.getItem('rag_settings_version')
  if (saved && version === SETTINGS_VER) {
    try { return JSON.parse(saved) } catch { /* fall through */ }
  }
  localStorage.setItem('rag_settings_version', SETTINGS_VER)
  return DEFAULTS
}

// ─── Pipeline loading stage hook ────────────────────────────────
function usePipelineStages(loading) {
  const [stage, setStage] = useState(0)
  const timers = useRef([])
  useEffect(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    if (!loading) { setStage(0); return }
    setStage(1)
    timers.current.push(setTimeout(() => setStage(2), 1300))
    timers.current.push(setTimeout(() => setStage(3), 2800))
    return () => timers.current.forEach(clearTimeout)
  }, [loading])
  return stage
}

// ─── Citation rendering ─────────────────────────────────────────
function AnswerText({ text, contexts, highlighted, onCite }) {
  if (!text) return null
  const declined = text.includes('not have enough information')
  const parts = text.split(/(\[[0-9,\s]+\])/g)

  return (
    <p className={`answer ${declined ? 'declined' : ''}`}>
      {parts.map((part, i) => {
        const match = part.match(/^\[([0-9,\s]+)\]$/)
        if (match) {
          const nums = match[1].split(',').map(s => s.trim())
          return (
            <span key={i} className="cite-group">
              [
              {nums.map((n, j) => {
                const ctx = contexts?.[parseInt(n) - 1]
                const id  = ctx?.citation_id ?? `c-${n}`
                return (
                  <span key={j}>
                    <button
                      type="button"
                      className={`cite-btn ${highlighted === id ? 'active' : ''}`}
                      onClick={() => onCite(id)}
                    >
                      {n}
                    </button>
                    {j < nums.length - 1 ? ', ' : ''}
                  </span>
                )
              })}
              ]
            </span>
          )
        }
        return part
      })}
    </p>
  )
}

// ─── Clean up raw extracted text for display ────────────────────
function cleanText(raw) {
  if (!raw) return ''
  return raw
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+/g, ' ')          // collapse horizontal whitespace
    .replace(/\n{3,}/g, '\n\n')       // max two consecutive newlines
    .replace(/- \n/g, '')             // remove hyphenated line-breaks
    .trim()
}

// ─── Source card ────────────────────────────────────────────────
function SourceCard({ ctx, index, highlighted, isExpanded, onToggle, onCite }) {
  const id = ctx.citation_id
  const r  = ctx.relevance_info || {}
  const isActive = highlighted === id
  const cleanBody = cleanText(ctx.text)

  // Auto-expand when activated via citation click
  useEffect(() => {
    if (isActive && !isExpanded) onToggle(id, true)
  }, [isActive]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      id={`src-${id}`}
      className={`src-card glass ${isActive ? 'active' : ''}`}
      onClick={() => { onToggle(id, !isExpanded); onCite(id) }}
    >
      <div className="src-card-head">
        <div className="src-num">{index + 1}</div>
        <div className="src-meta">
          <h3 title={id}>{id}</h3>
          <div className="pills">
            {r.cohere_rerank_score != null && (
              <span className="pill cohere">Rerank {r.cohere_rerank_score.toFixed(3)}</span>
            )}
            {r.distance != null && (
              <span className="pill vector">Dist {r.distance.toFixed(3)}</span>
            )}
            {r.bm25_score != null && (
              <span className="pill bm25">BM25 {r.bm25_score.toFixed(2)}</span>
            )}
          </div>
        </div>
        <span className={`chevron ${isExpanded ? 'open' : ''}`}>▶</span>
      </div>
      {isExpanded && (
        <div className="src-body">
          <pre className="src-text">{cleanBody}</pre>
        </div>
      )}
    </div>
  )
}

// ─── Main App ───────────────────────────────────────────────────
export default function App() {
  const [query,        setQuery]        = useState('')
  const [loading,      setLoading]      = useState(false)
  const [apiOnline,    setApiOnline]    = useState(null)
  const [response,     setResponse]     = useState(null)
  const [error,        setError]        = useState(null)
  const [highlighted,  setHighlighted]  = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [settings,     setSettings]     = useState(loadSettings)
  const [expandedCards,setExpandedCards]= useState(() => new Set())
  const [queryMs,      setQueryMs]      = useState(null)
  const [showReferences, setShowReferences] = useState(false)
  const pipelineStage = usePipelineStages(loading)

  // Persist settings
  useEffect(() => {
    localStorage.setItem('rag_settings', JSON.stringify(settings))
  }, [settings])

  // Health check
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('http://localhost:8000/api/health')
        setApiOnline((await r.json()).status === 'ok')
      } catch { setApiOnline(false) }
    }
    check()
    const t = setInterval(check, 10_000)
    return () => clearInterval(t)
  }, [])

  // Query handler
  const handleSearch = async (e, overrideQuery) => {
    if (e) e.preventDefault()
    const q = (overrideQuery ?? query).trim()
    if (!q) return

    setLoading(true); setError(null); setResponse(null); setHighlighted(null)
    setExpandedCards(new Set()); setQueryMs(null)
    const t0 = Date.now()

    try {
      const res = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query:       q,
          data_dir:    settings.data_dir,
          mode:        'hybrid',
          re_rank:     settings.re_rank,
          k:           parseInt(settings.k),
          pool_size:   parseInt(settings.pool_size),
          answer_mode: settings.answer_mode,
          gemini_key:  settings.gemini_key || null,
          cohere_key:  settings.cohere_key || null,
        }),
      })
      if (!res.ok) {
        const e = await res.json()
        throw new Error(e.detail || `HTTP ${res.status}`)
      }
      setQueryMs(Date.now() - t0)
      setResponse(await res.json())
    } catch (err) {
      setError(err.message || 'Failed to connect to RAG backend.')
    } finally {
      setLoading(false)
    }
  }

  const handleCite = (id) => {
    setShowReferences(true)
    setHighlighted(id)
    setTimeout(() => {
      const el = document.getElementById(`src-${id}`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }

  const handleToggleCard = (id, open) => {
    setExpandedCards(prev => {
      const next = new Set(prev)
      if (open) next.add(id)
      else next.delete(id)
      return next
    })
  }

  const patch = (key, val) => setSettings(p => ({ ...p, [key]: val }))



  const statusClass = apiOnline === true ? 'online' : apiOnline === false ? 'offline' : 'pending'
  const statusText  = apiOnline === true ? 'API Connected' : apiOnline === false ? 'API Offline' : 'Connecting…'

  const keysConfigured = !!settings.gemini_key && !!settings.cohere_key
  const missingKeys = []
  if (!settings.gemini_key)  missingKeys.push('Gemini')
  if (!settings.cohere_key)  missingKeys.push('Cohere')

  return (
    <div className="workspace">
      {/* ─── Left Navigation Sidebar ──────────────────────── */}
      <nav className="nav-sidebar">
        <div className="nav-sidebar-top">
          <button
            type="button"
            className={`nav-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(prev => !prev)}
            title="RAG Parameters & Settings"
          >
            ☰
          </button>
          
          <button
            type="button"
            className="nav-btn primary-btn"
            disabled
            style={{ opacity: 0.3, cursor: 'not-allowed' }}
            title="New Query (Disabled)"
          >
            +
          </button>
        </div>
        
        <div className="nav-sidebar-bottom">
          <button
            type="button"
            className="nav-btn"
            title="Chat History (Coming Soon)"
            disabled
            style={{ opacity: 0.3, cursor: 'not-allowed' }}
          >
            💬
          </button>
          
          <button
            type="button"
            className={`nav-btn ${showSettings ? 'active' : ''}`}
            disabled
            style={{ opacity: 0.3, cursor: 'not-allowed' }}
            title="RAG Parameters & Settings (Disabled)"
          >
            ⚙
          </button>
        </div>
      </nav>

      {/* ─── Main Content Area ────────────────────────────── */}
      <main className="main-content">
        <div className="center-container">
          
          {/* Search Area Wrapper to keep Search Container width constant */}
          <div className="search-area-wrapper" style={{ width: '100%', maxWidth: '820px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}>
            {/* Top Status Header */}
            <div className="top-header">
              <div className={`status-pill`}>
                <span className={`status-dot ${statusClass}`} />
                {statusText}
              </div>
              <div className="user-avatar" title="User Settings" onClick={() => setShowSettings(true)} />
            </div>

            {/* Hero Title */}
            <h1 className="hero-title">What can I help with?</h1>

            {/* Search Card Container */}
            <div className="search-card">
              <form className="search-row" onSubmit={handleSearch}>
                <div className="search-wrap">
                  <textarea
                    className="search-input"
                    placeholder="Ask the research corpus..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSearch(e)
                      }
                    }}
                    disabled={loading}
                    rows={1}
                  />
                </div>
                
                <div className="search-toolbar">
                  <div className="toolbar-left">
                    <button
                      type="button"
                      className={`toolbar-btn ${showReferences ? 'active' : ''}`}
                      onClick={() => setShowReferences(prev => !prev)}
                    >
                      📄 References {response?.contexts?.length ? `(${response.contexts.length})` : ''}
                    </button>
                  </div>
                  
                  <button
                    type="submit"
                    className="search-btn"
                    disabled={loading || !query.trim()}
                    title="Submit Query"
                  >
                    ↑
                  </button>
                </div>
              </form>
            </div>

            {/* Keys Missing Warning Banner */}
            {!keysConfigured && (
              <div className="keys-banner">
                <div className="keys-banner-left">
                  <span className="keys-banner-icon">🔑</span>
                  <div>
                    <strong>API keys required</strong>
                    <span className="keys-banner-sub">
                      {missingKeys.join(' & ')} {missingKeys.length > 1 ? 'keys are' : 'key is'} not configured in RAG settings drawer.
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  className="keys-banner-btn"
                  onClick={() => setShowSettings(true)}
                >
                  Configure Keys →
                </button>
              </div>
            )}
          </div>

          {/* Main Grid: Output and references section */}
          {(loading || response || error) && (
            <div className="main-grid">
              
              {/* Left Column (Answer) */}
              <div className="left-col">
                <div className="response-card">
                  {loading && (() => {
                    const stage = pipelineStage
                    const isGen = settings.answer_mode !== 'extractive'
                    const stages = [
                      { id: 1, label: 'Retrieving' },
                      { id: 2, label: isGen ? 'Re-ranking' : 'Re-ranking' },
                      { id: 3, label: isGen ? 'Generating' : 'Extracting' },
                    ]
                    return (
                      <div className="skel-wrap">
                        {/* Progress bar */}
                        <div className="load-progress-track">
                          <div className="load-progress-fill" />
                        </div>

                        {/* Stage pills */}
                        <div className="load-stages">
                          {stages.map((s, idx) => {
                            const status = stage > s.id ? 'done' : stage === s.id ? 'active' : ''
                            return (
                              <>
                                <div key={s.id} className={`load-stage ${status}`}>
                                  <span className="load-stage-dot" />
                                  {status === 'done' ? '✓ ' : ''}{s.label}
                                </div>
                                {idx < stages.length - 1 && (
                                  <span key={`sep-${s.id}`} className="load-sep">›</span>
                                )}
                              </>
                            )
                          })}
                        </div>

                        {/* Skeleton lines */}
                        <div className="skel-lines">
                          <div className="skel shimmer w-90" style={{ height: 15 }} />
                          <div className="skel shimmer w-85" />
                          <div className="skel shimmer w-90" />
                          <div className="skel shimmer w-75" />
                          <div className="skel shimmer w-60" />
                        </div>

                        <p className="loading-label">
                          Searching your research corpus
                          <span className="loading-dots">
                            <span /><span /><span />
                          </span>
                        </p>
                      </div>
                    )
                  })()}

                  {error && (
                    <div className="error-box">
                      <div className="err-icon">⚠️</div>
                      <div>
                        <h3>Query Failed</h3>
                        <p>{error}</p>
                        <small>Make sure the API server is running on port 8000 and your keys are correct.</small>
                      </div>
                    </div>
                  )}

                  {response && (
                    <>
                      <div className="resp-meta">
                        <span className={`badge ${response.answer.includes('not have enough information') ? 'badge-declined' : 'badge-grounded'}`}>
                          {response.answer.includes('not have enough information') ? '✗ Declined' : '✓ Grounded'}
                        </span>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                          <button
                            type="button"
                            className="badge"
                            onClick={() => setShowReferences(prev => !prev)}
                            style={{
                              background: showReferences ? 'var(--primary-glow)' : 'rgba(0,0,0,0.04)',
                              border: showReferences ? '1px solid var(--border-focus)' : '1px solid var(--border)',
                              color: 'var(--text-1)',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              padding: '4px 10px',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: '11px',
                              fontWeight: '600',
                              fontFamily: 'var(--font-sans)',
                              letterSpacing: 'normal'
                            }}
                          >
                            📄 {showReferences ? 'Hide Sources' : `Show Sources (${response.contexts?.length ?? 0})`}
                          </button>
                          {queryMs != null && (
                            <span className="badge badge-time" title="Total round-trip time">⏱ {(queryMs/1000).toFixed(1)}s</span>
                          )}
                        </div>
                      </div>
                      <AnswerText
                        text={response.answer}
                        contexts={response.contexts}
                        highlighted={highlighted}
                        onCite={handleCite}
                      />
                    </>
                  )}
                </div>
              </div>

              {/* Source References (displayed below answer) */}
              {showReferences && response && (
                <div className="references-section">
                  <div className="sources-header">
                    <h2>Source References</h2>
                    <span className="count-badge">
                      {response.contexts?.length ?? 0} matched
                    </span>
                  </div>

                  <div className="sources-grid">
                    {response.contexts?.map((ctx, i) => (
                      <SourceCard
                        key={ctx.citation_id + i}
                        ctx={ctx}
                        index={i}
                        highlighted={highlighted}
                        isExpanded={expandedCards.has(ctx.citation_id)}
                        onToggle={handleToggleCard}
                        onCite={handleCite}
                      />
                    ))}
                  </div>
                </div>
              )}

            </div>
          )}

          {/* Footer disclaimer removed */}

        </div>
      </main>

      {/* ─── Settings Drawer ──────────────────────────────── */}
      {showSettings && (
        <div className="drawer-backdrop" onClick={() => setShowSettings(false)}>
          <aside className="drawer glass" onClick={e => e.stopPropagation()}>
            <div className="drawer-head">
              <h2>RAG Parameters</h2>
              <button type="button" className="close-btn" onClick={() => setShowSettings(false)}>✕</button>
            </div>

            <div className="drawer-body">
              {/* Answer Mode */}
              <div className="setting">
                <span className="setting-label">Answer Mode</span>
                <div className="mode-tabs">
                  {[
                    { val: 'extractive', label: 'Extractive' },
                    { val: 'gemini',     label: 'Gemini LLM' },
                  ].map(({ val, label }) => (
                    <button
                      key={val}
                      type="button"
                      className={`mode-tab ${settings.answer_mode === val ? 'on' : ''}`}
                      onClick={() => patch('answer_mode', val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <small style={{ color: 'var(--text-3)', fontSize: 10 }}>
                  {settings.answer_mode === 'extractive'
                    ? 'Local · no API key needed · returns exact sentences from sources'
                    : 'Requires Gemini API key · synthesises a fluent answer'}
                </small>
              </div>

              {/* Cohere Re-rank toggle */}
              <div className="setting toggle-row">
                <div className="toggle-info">
                  <span className="setting-label">Cohere Re-ranking</span>
                  <small>Cross-encoder second-stage re-ranking</small>
                </div>
                <input
                  type="checkbox"
                  className="toggle"
                  checked={settings.re_rank}
                  onChange={e => patch('re_rank', e.target.checked)}
                />
              </div>

              {/* Top-K */}
              <div className="setting">
                <div className="slider-row">
                  <span className="setting-label">Top K</span>
                  <span className="slider-val">{settings.k}</span>
                </div>
                <input
                  type="range" min={1} max={10} step={1}
                  className="slider"
                  value={settings.k}
                  onChange={e => patch('k', parseInt(e.target.value))}
                />
              </div>

              {/* Pool Size */}
              <div className="setting">
                <div className="slider-row">
                  <span className="setting-label">Initial Pool Size</span>
                  <span className="slider-val">{settings.pool_size}</span>
                </div>
                <input
                  type="range" min={5} max={40} step={5}
                  className="slider"
                  value={settings.pool_size}
                  disabled={!settings.re_rank}
                  onChange={e => patch('pool_size', parseInt(e.target.value))}
                />
              </div>

              {/* Data Dir */}
              <div className="setting">
                <span className="setting-label">Data Directory</span>
                <input
                  type="text"
                  className="txt-input"
                  value={settings.data_dir}
                  onChange={e => patch('data_dir', e.target.value)}
                  placeholder="data/llm_ai_2023_2026"
                />
              </div>

              <hr className="divider" />

              {/* API Keys */}
              <div className="setting">
                <p className="section-head">API Key Overrides</p>
                <p className="section-desc">Stored in browser localStorage only. Sent directly to the local server.</p>
              </div>

              <div className="setting key-field">
                <label>Gemini API Key</label>
                <input
                  type="password"
                  className="txt-input"
                  value={settings.gemini_key}
                  onChange={e => patch('gemini_key', e.target.value)}
                  placeholder="AIzaSy…"
                />
              </div>

              <div className="setting key-field">
                <label>Cohere API Key</label>
                <input
                  type="password"
                  className="txt-input"
                  value={settings.cohere_key}
                  onChange={e => patch('cohere_key', e.target.value)}
                  placeholder="Enter Cohere key…"
                />
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}

