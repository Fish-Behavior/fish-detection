import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import {
  COLORS,
  effectiveSegments,
  emptyEdits,
  frameAt,
  frameIndexAt,
  hasEdits,
  staleFor,
  validateFrame,
  validateInterval,
  validateReviewer,
  validateScene,
} from './model.ts'
import type {
  Behavior,
  Box,
  Edits,
  LabelEdit,
  Point,
  Scene,
  SessionData,
  SessionSummary,
  ReviewData,
} from './model.ts'
import { ApiError, askChat, loadSession, listSessions, loadOverlay, saveReview, rerun, predict, startProcessing, processingStatus } from './api.ts'
import { Card, Icon } from './ui.tsx'
import VideoPanel from './VideoPanel.tsx'
import type { EditTool } from './VideoPanel.tsx'
import Editors, { LabelEditor } from './Editors.tsx'
import { Ethogram, StateSummary, Traces } from './Charts.tsx'
import Chat from './Chat.tsx'
import ReviewTools from './ReviewTools.tsx'

function Empty({ title, message }: { title: string; message?: string }) {
  return (
    <main className="empty-page">
      <span className="brand-mark">
        <Icon name="fish" size={26} />
      </span>
      <h1>{title}</h1>
      {message && <p className="muted">{message}</p>}
    </main>
  )
}

// Renders nothing but this shell until the backend (src/api.ts) provides a session.
export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [selected, setSelected] = useState('')
  const generation = useRef(0)
  const pendingCount = useRef(0)
  const [load, setLoad] = useState<
    | { status: 'loading' }
    | { status: 'error'; stage: 'connection' | 'inventory' | 'recording'; message: string }
    | { status: 'ready'; session: SessionData | null }
  >({ status: 'loading' })
  const reload = async (id = selected) => {
    if (pendingCount.current && !window.confirm('Discard staged behavior edits and reload this recording?')) return
    const current = ++generation.current
    setLoad({ status: 'loading' })
    let stage: 'inventory' | 'recording' = 'inventory'
    try {
      const rows = await listSessions()
      if (current !== generation.current) return
      const next = rows.some((r) => r.video_id === id) ? id : rows[0]?.video_id ?? ''
      setSessions(rows); setSelected(next)
      stage = 'recording'
      const session = next ? await loadSession(next) : null
      if (current !== generation.current) return
      setLoad({ status: 'ready', session })
    } catch (e) {
      if (current === generation.current) setLoad({ status: 'error', stage: e instanceof ApiError && e.code === 'offline' ? 'connection' : stage, message: e instanceof Error ? e.message : 'The request failed.' })
    }
  }
  useEffect(() => { void reload(); return () => { generation.current++ } }, [])
  return <>
    <div className={`backend-toolbar${load.status === 'ready' && load.session ? ' main-shell' : ''}`}>
      <button onClick={() => void reload()}>Reload from backend</button>
    </div>
    {load.status === 'loading' ? <Empty title="Loading…" /> : load.status === 'error'
      ? <Empty title={load.stage === 'connection' ? 'Could not connect to the API' : load.stage === 'inventory' ? 'Could not load the recording list' : 'Could not load this recording'} message={load.message} /> : !load.session
      ? <Empty title="No recordings yet" message="Set PDS_VIDEO_DIR to your recordings folder and reload." />
      : <Workspace key={load.session.video_id + ':' + load.session.review?.revision} session={load.session} sessions={sessions} selectSession={(id) => void reload(id)} refresh={() => void reload()} onPendingChange={(count) => { pendingCount.current = count }} />}
  </>
}

function Workspace({ session: initialSession, sessions, selectSession, refresh, onPendingChange }: { session: SessionData; sessions: SessionSummary[]; selectSession: (id: string) => void; refresh: () => void; onPendingChange: (count: number) => void }) {
  const [session, setSession] = useState(initialSession)
  const [review, setReview] = useState<ReviewData>(session.review!)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const initialScene = (): Scene => structuredClone(session.scene)
  const firstState: Behavior = session.segments[0]?.state ?? 'Undetermined'
  const firstRange = (): [number, number] => [0, Math.min(1, session.duration)]
  const [tab, setTab] = useState<'dashboard' | 'review'>('dashboard')
  const [chatOpen, setChatOpen] = useState(false)
  const [edits, setEdits] = useState<Edits>(() => structuredClone(session.review?.edits ?? emptyEdits())),
    [reviewer, setReviewer] = useState(session.review?.reviewer ?? '')
  const [time, setTime] = useState(0),
    [duration, setDuration] = useState(session.duration)
  const [tool, setTool] = useState<EditTool>('inspect'),
    [keypoint, setKeypoint] = useState('snout')
  const [sceneDraft, setSceneDraft] = useState(() => structuredClone(edits.scene ?? session.scene)),
    [frameDraft, setFrameDraft] = useState(() =>
      frameAt(session, 0, edits),
    )
  const [range, setRange] = useState<[number, number]>(firstRange),
    [label, setLabel] = useState<Behavior>(firstState)
  const [pendingLabels, setPendingLabels] = useState<LabelEdit[]>([])
  const [compound, setCompound] = useState(
      edits.finalResult?.compound ?? session.predictions?.predicted ?? '',
    ),
    [dose, setDose] = useState(edits.finalResult?.dose ?? '')
  const [reviewStatus, setReviewStatus] = useState(review.decision ?? (hasEdits(edits) ? 'EDITED' : session.has_tracking ? 'PROCESSED_AUTO' : 'NOT_PROCESSED')),
    [filter, setFilter] = useState('all')
  const [notice, setNotice] = useState(''),
    [error, setError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const hasTracking = session.has_tracking ?? session.overlay.t.length > 0,
    frameIndex = frameIndexAt(session, time)
  const stale = session.stale ?? staleFor(edits)
  const savedSegments = session.reviewed_segments ?? effectiveSegments(session.segments, edits.labels)
  const segments = tab === 'review' ? effectiveSegments(savedSegments, pendingLabels) : savedSegments
  useEffect(() => {
    onPendingChange(pendingLabels.length)
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    if (pendingLabels.length) window.addEventListener('beforeunload', warn)
    return () => { onPendingChange(0); window.removeEventListener('beforeunload', warn) }
  }, [pendingLabels.length, onPendingChange])
  const predictions = session.predictions
  const visibleSessions = sessions.filter((item) =>
    filter === 'all' || (filter === 'edited' && item.edited) ||
    (filter === 'needs-review' && !['ACCEPTED', 'REJECTED'].includes(item.review_status)),
  )
  const predictedProbability = predictions
    ? predictions[`p:${predictions.predicted}`]
    : NaN
  const current =
    segments.find((s) => s.start_s <= time && s.end_s > time) ?? segments.at(-1)
  // Only preview a relabel that would actually change something in the chosen range.
  const labelDraft =
    tab === 'review' &&
    range.every(Number.isFinite) &&
    range[0] < range[1] &&
    segments.some(
      (s) => s.start_s < range[1] && s.end_s > range[0] && s.state !== label,
    )
  const editCount =
    Object.keys(edits.frames).length +
    edits.labels.length +
    Number(!!edits.scene) +
    Number(!!edits.finalResult)
  useLayoutEffect(() => {
    setFrameDraft(frameAt(session, frameIndex, edits))
  }, [frameIndex, edits.frames, session.overlay])
  const windowStart = Math.max(0, Math.floor(time / 25) * 25 - 2)
  useEffect(() => {
    let alive = true
    loadOverlay(session.video_id, windowStart, Math.min(windowStart + 30, session.duration)).then(
      (overlay) => { if (alive) setSession((s) => ({ ...s, overlay })) },
      (e) => { if (alive) setError(e.message) },
    )
    return () => { alive = false }
  }, [session.video_id, windowStart])
  const seek = useCallback(
    (t: number) => {
      if (!Number.isFinite(t)) return
      const next = Math.max(0, Math.min(t, duration))
      if (videoRef.current) videoRef.current.currentTime = next
      setTime(next)
    },
    [duration],
  )
  const announce = (message: string) => {
    setError('')
    setNotice(message)
  }
  const apply = async (action: (edits: Edits) => Edits, message: string, decision: ReviewData['decision'] = null) => {
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const next = action(structuredClone(edits))
      const saved = await saveReview(session, review, next, reviewer.trim(), decision)
      setReview(saved); setEdits(saved.edits)
      setReviewStatus(saved.decision ?? (hasEdits(saved.edits) ? 'EDITED' : hasTracking ? 'PROCESSED_AUTO' : 'NOT_PROCESSED'))
      const changed = ['scene', 'frames', 'labels'].some((key) => JSON.stringify(next[key as keyof Edits]) !== JSON.stringify(edits[key as keyof Edits]))
      if (changed) setSession((s) => ({ ...s, reviewed_segments: null, stale: staleFor(next) }))
      try {
        const fresh = await loadSession(session.video_id)
        if (fresh) setSession({ ...fresh, overlay: session.overlay })
      } catch (e) {
        announce(`${message} Reload the session to refresh derived data: ${(e as Error).message}`)
        return true
      }
      announce(message)
      return true
    } catch (e) {
      setNotice('')
      setError(
        e instanceof Error ? e.message : 'The correction could not be applied.',
      )
    } finally { busyRef.current = false; setBusy(false) }
  }
  const reset = async () => {
    if (!await apply(() => emptyEdits(), 'Corrections reset to the automatic baseline.')) return
    setPendingLabels([])
    videoRef.current?.pause()
    setTime(0)
    setTool('inspect')
    setSceneDraft(initialScene())
    setFrameDraft(frameAt(session, 0, emptyEdits()))
    setRange(firstRange())
    setCompound(session.predictions?.predicted ?? '')
    setDose('')
    setFilter('all')
  }
  const sketch = (kind: EditTool, value: Point | Box) => {
    const p = value as Point
    if (kind === 'waterline') setSceneDraft((s) => ({ ...s, waterline: p[1] }))
    if (kind === 'roi') setSceneDraft((s) => ({ ...s, roi: value as Box }))
    if (kind === 'point')
      setFrameDraft((f) => ({ ...f, x: p[0], y: p[1], detected: true }))
    if (kind === 'box') setFrameDraft((f) => ({ ...f, box: value as Box }))
    if (kind === 'keypoint')
      setFrameDraft((f) =>
        f.keypoints[keypoint]
          ? {
              ...f,
              keypoints: {
                ...f.keypoints,
                [keypoint]: [p[0], p[1], f.keypoints[keypoint][2]],
              },
            }
          : f,
      )
  }
  const nudge = (kind: EditTool, dx: number, dy: number) => {
    const { width, height } = session.overlay,
      clamp = (v: number, max: number) => Math.max(0, Math.min(max, v)),
      shift = (b: Box): Box => {
        const x = Math.max(-b[0], Math.min(width - b[2], dx)),
          y = Math.max(-b[1], Math.min(height - b[3], dy))
        return [b[0] + x, b[1] + y, b[2] + x, b[3] + y]
      }
    if (kind === 'waterline')
      setSceneDraft((s) => ({
        ...s,
        waterline: clamp((s.waterline ?? 0) + dy, height),
      }))
    if (kind === 'roi') setSceneDraft((s) => ({ ...s, roi: shift(s.roi) }))
    if (kind === 'box')
      setFrameDraft((f) => (f.box ? { ...f, box: shift(f.box) } : f))
    if (kind === 'point')
      setFrameDraft((f) => ({
        ...f,
        x: clamp((f.x ?? width / 2) + dx, width),
        y: clamp((f.y ?? height / 2) + dy, height),
        detected: true,
      }))
    if (kind === 'keypoint')
      setFrameDraft((f) => {
        if (!f.keypoints[keypoint]) return f
        const [x, y, c] = f.keypoints[keypoint]
        return {
          ...f,
          keypoints: {
            ...f.keypoints,
            [keypoint]: [clamp(x + dx, width), clamp(y + dy, height), c],
          },
        }
      })
  }
  const saveLabels = () =>
    apply((e) => {
      validateInterval(range[0], range[1], session.duration)
      return {
        ...e,
        labels: [
          ...e.labels,
          {
            start_s: range[0],
            end_s: range[1],
            state: label,
            reviewer: reviewer.trim(),
          },
        ],
      }
    }, 'Behavior correction saved. Timeline and totals updated; feature/prediction results are stale.')
  const stageLabels = () => {
    try {
      validateReviewer(reviewer)
      validateInterval(range[0], range[1], session.duration)
      setPendingLabels((pending) => [...pending, { start_s: range[0], end_s: range[1], state: label, reviewer: reviewer.trim() }])
      announce('Behavior edit staged. Save the batch to persist it.')
    } catch (e) { setError((e as Error).message) }
  }
  const saveStagedLabels = async () => {
    if (await apply((e) => ({ ...e, labels: [...e.labels, ...pendingLabels] }), 'Staged behavior edits saved. Features and predictions are stale.')) setPendingLabels([])
  }
  const resetLabels = async () => {
    if (await apply((e) => ({ ...e, labels: [] }), 'Automatic behavior labels restored.')) setPendingLabels([])
  }
  const recalculate = async () => {
    if (pendingLabels.length) { setError('Save or discard staged labels before rerunning analysis.'); return }
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const updated = await rerun(session, review, reviewer.trim())
      setSession({ ...updated, overlay: session.overlay }); setReview(updated.review!); setEdits(updated.review!.edits)
      announce('Measurements and reviewed labels recalculated. Compound predictions remain stale until a model runs on the corrected data.')
    } catch (e) { setNotice(''); setError((e as Error).message) }
    finally { busyRef.current = false; setBusy(false) }
  }
  const [processing, setProcessing] = useState(session.processing)
  useEffect(() => {
    if (!['queued', 'running'].includes(processing?.status ?? '')) return
    let alive = true
    const poll = setInterval(() => {
      processingStatus(session.video_id).then((value) => {
        if (!alive) return
        setProcessing(value)
        if (value.status === 'processed') refresh()
        if (['failed', 'interrupted'].includes(value.status)) setError(value.message)
      }, (e) => { if (alive) { setError(e.message); setProcessing({ status: 'interrupted', message: e.message }) } })
    }, 2000)
    return () => { alive = false; clearInterval(poll) }
  }, [processing?.status, session.video_id])
  const processVideo = async () => {
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      setProcessing(await startProcessing(session.video_id, reviewer.trim()))
      announce('Analysis queued. Processing status updates automatically.')
    } catch (e) { setNotice(''); setError((e as Error).message) }
    finally { busyRef.current = false; setBusy(false) }
  }
  const predictCompound = async () => {
    if (pendingLabels.length) { setError('Save or discard staged labels before running the compound model.'); return }
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const updated = await predict(session, review, reviewer.trim())
      setSession({ ...updated, overlay: session.overlay }); setReview(updated.review!)
      announce('Compound probabilities recalculated by the configured DCS model.')
    } catch (e) { setNotice(''); setError((e as Error).message) }
    finally { busyRef.current = false; setBusy(false) }
  }
  const changeTab = (next: 'dashboard' | 'review') => {
    setTab(next)
    setTool('inspect')
  }
  const results = (
    <Card
      title="Compound prediction"
      eyebrow="Drug results"
      accessory={
        stale.predictions && <span className="badge warning">Stale</span>
      }
    >
      {predictions ? (
        <>
          <div className="prediction-lead">
            <span className="muted small">Automatic model result</span>
            <div>
              <strong>{predictions!.predicted.replaceAll('_', ' ')}</strong>
              <span>
                {Number.isFinite(predictedProbability)
                  ? Math.round(predictedProbability * 100)
                  : '—'}
                {Number.isFinite(predictedProbability) && (
                  <span className="percent">%</span>
                )}
              </span>
            </div>
          </div>
          <div className="probabilities">
            {Object.entries(predictions!)
              .filter(([k]) => k.startsWith('p:'))
              .sort(([, a], [, b]) => Number(b) - Number(a))
              .map(([key, value], i) => (
                <div className="probability" key={key}>
                  <div>
                    <span>{key.slice(2).replaceAll('_', ' ')}</span>
                    <b>{(Number(value) * 100).toFixed(0)}%</b>
                  </div>
                  <div className="bar-track">
                    <div
                      style={{
                        width: `${Number(value) * 100}%`,
                        background:
                          ['#2c70b6', '#8ba8c8', '#c7d6e4'][i] ?? '#c7d6e4',
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </>
      ) : (
        <p className="muted small">
          No model result is available for this session.
        </p>
      )}
      <div className="dose-status">
        <span>Model dose estimate</span>
        <b>Unavailable</b>
      </div>
      {edits.finalResult && (
        <div className="manual-result">
          <span className="badge manual">Manual final result</span>
          <h3>{edits.finalResult.compound}</h3>
          <p>
            {edits.finalResult.dose || 'Dose not specified'} · by{' '}
            {edits.finalResult.reviewer}
          </p>
        </div>
      )}
      <details className="result-editor">
        <summary>Override final drug / dose</summary>
        <div className="result-fields">
          <label className="field">
            Reviewer
            <input
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="Your name"
              maxLength={100}
            />
          </label>
          <label className="field">
            Final compound
            <input
              value={compound}
              onChange={(e) => setCompound(e.target.value)}
              maxLength={100}
            />
          </label>
          <label className="field">
            Manual dose (optional, include unit)
            <input
              value={dose}
              onChange={(e) => setDose(e.target.value)}
              maxLength={100}
              placeholder="e.g. 0.5 mM"
            />
          </label>
          <div className="button-row">
            <button
              className="primary"
              onClick={() =>
                apply((e) => {
                  if (!compound.trim())
                    throw new Error('Enter a final compound.')
                  return {
                    ...e,
                    finalResult: {
                      compound: compound.trim(),
                      dose: dose.trim(),
                      reviewer: reviewer.trim(),
                    },
                  }
                }, 'Manual final result saved. Automatic probabilities are unchanged.')
              }
            >
              Apply final result
            </button>
            <button
              onClick={async () => {
                if (await apply((e) => ({ ...e, finalResult: null }), 'Manual final result removed.')) {
                  setCompound(session.predictions?.predicted ?? ''); setDose('')
                }
              }}
            >
              Restore model result
            </button>
          </div>
          <p className="small muted">
            A manual decision does not change model probabilities.
          </p>
        </div>
      </details>
      <button
        className="rerun-button"
        onClick={() => void recalculate()}
      >
        <Icon name="reset" />
        Rerun affected stages
      </button>
      <button className="rerun-button" onClick={() => void predictCompound()}>Run compound model</button>
    </Card>
  )

  return (
    <fieldset disabled={busy || ['queued', 'running'].includes(processing?.status ?? '')} className="app-shell workspace-controls" aria-busy={busy}>
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault()
            changeTab('dashboard')
          }}
        >
          <span className="brand-mark">
            <Icon name="fish" size={26} />
          </span>
          <span>
            FishLab<small>BEHAVIOR ANALYSIS</small>
          </span>
        </a>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Workspace views">
          <button
            className={tab === 'dashboard' ? 'active' : ''}
            onClick={() => changeTab('dashboard')}
          >
            <Icon name="dashboard" />
            Dashboard
          </button>
          <button
            className={tab === 'review' ? 'active' : ''}
            onClick={() => changeTab('review')}
          >
            <Icon name="review" />
            Review
            {editCount > 0 && <span className="nav-count">{editCount}</span>}
          </button>
        </nav>
        <div className="sidebar-session">
          <div className="sidebar-label">
            <span>SESSIONS</span>
            <span>{sessions.length}</span>
          </div>
          <label className="sr-only" htmlFor="session-filter">
            Filter sessions
          </label>
          <select
            id="session-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All sessions</option>
            <option value="edited">With corrections</option>
            <option value="needs-review">Needs review</option>
          </select>
          <div className="session-list" aria-label="Video inventory">
            {visibleSessions.map((item) => (
              <button key={item.video_id}
                className={`session-item ${item.video_id === session.video_id ? 'selected' : ''}`}
                aria-current={item.video_id === session.video_id ? 'true' : undefined}
                onClick={() => item.video_id !== session.video_id && selectSession(item.video_id)}>
                <span className={`session-dot ${item.processing_status}`} />
                <span className="session-item-text">
                  <b>{item.video_id}</b>
                  <small>{item.name}</small>
                  <span className="session-status">{item.processing_status}{item.edited ? ' · edited' : ''}</span>
                </span>
              </button>
            ))}
            {visibleSessions.length === 0 && <p className="small muted">No matching sessions.</p>}
          </div>
        </div>
        <div className="sidebar-footer">
          <span className="live-dot" />
          <div>
            <b>FishLab</b>
            <small>Behavior analysis</small>
          </div>
          <span className="version">v0.1</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            <b>Workspace</b>
            <span className="crumb">/</span>
            {tab === 'dashboard' ? 'Dashboard' : 'Review'}
          </span>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <p className="eyebrow">ZEBRAFISH · SESSION WORKSPACE</p>
              <h1>{tab === 'dashboard' ? 'Dashboard' : 'Review'}</h1>
              <p className="subtitle">
                {tab === 'dashboard'
                  ? 'Follow the pipeline, explore behavior, and review every result.'
                  : 'Inspect the evidence and keep your corrections alongside the original.'}
              </p>
            </div>
            <button className="reset-button" onClick={reset}>
              <Icon name="reset" />
              Reset edits
            </button>
          </div>
          <div className="session-heading">
            <div className="session-heading-main">
              <span className="session-icon">
                <Icon name="fish" size={23} />
              </span>
              <div>
                <h2>{session.video_id}</h2>
                <p>
                  {session.name} · {session.duration.toFixed(1)} s
                </p>
              </div>
            </div>
            <span
              className={`badge ${reviewStatus === 'ACCEPTED' ? 'success' : 'neutral'}`}
            >
              {reviewStatus.replaceAll('_', ' ')}
            </span>
          </div>
          <ol className="pipeline" aria-label="Pipeline stages">
            {['Scene', 'Tracking', 'Features', 'Behavior', 'Prediction'].map(
              (stage, i) => {
                const done = [
                  true,
                  hasTracking,
                  session.measurements.length > 0,
                  segments.length > 0,
                  !!predictions,
                ][i]
                const s =
                  i === 2
                    ? stale.measurements
                    : i === 3
                      ? stale.labels
                      : i === 4
                        ? stale.predictions
                        : false
                return (
                  <li key={stage} className={s ? 'stale' : ''}>
                    <span className="stage-number">
                      {done && !s ? (
                        <Icon name="check" size={14} />
                      ) : (
                        `0${i + 1}`
                      )}
                    </span>
                    <div>
                      <b>{stage}</b>
                      <small>
                        {s
                          ? 'Needs rerun'
                          : done
                            ? [
                                'Scene defined',
                                'Track loaded',
                                'Metrics computed',
                                'Labels assigned',
                                'Output ready',
                              ][i]
                            : 'Not available'}
                      </small>
                    </div>
                    {i < 4 && <span className="stage-connector">›</span>}
                  </li>
                )
              },
            )}
          </ol>
          <div className="notice-slot" aria-live="polite">
            {busy && <p className="notice">Saving to backend…</p>}
            {session.warnings?.map((w) => <p key={w} className="inline-error">{w}</p>)}
            {processing && processing.status !== 'idle' && <p className="notice">Analysis: {processing.status} {processing.message}</p>}
            {!hasTracking && <div className="processing-controls">
              <label>Reviewer <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} maxLength={100} /></label>
              <button onClick={() => void processVideo()} disabled={!session.processable}>Run analysis</button>
              {!session.processable && <p className="muted">No matched trial. Run prepds catalog and resolve the workbook match before analysis.</p>}
            </div>}
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            {notice && (
              <p className="notice">
                <Icon name="info" size={16} />
                {notice}
                <button
                  className="icon-button"
                  onClick={() => setNotice('')}
                  aria-label="Dismiss message"
                >
                  <Icon name="close" size={14} />
                </button>
              </p>
            )}
          </div>
          <div
            className={`workspace-grid ${tab === 'review' ? 'review-grid' : ''}`}
          >
            <div className="workspace-main">
              <VideoPanel
                reviewMode={tab === 'review'}
                markRange={(boundary) => setRange((r) => boundary === 'start' ? [time, r[1]] : [r[0], time])}
                session={session}
                edits={edits}
                src={session.video_url}
                title={session.name}
                time={time}
                duration={duration}
                setTime={setTime}
                setDuration={setDuration}
                videoRef={videoRef}
                seek={seek}
                tool={tool}
                keypoint={keypoint}
                sceneDraft={sceneDraft}
                frameDraft={frameDraft}
                sketch={sketch}
                nudge={nudge}
              />
              {tab === 'review' && pendingLabels.length > 0 && <p className="notice" role="status">Previewing {pendingLabels.length} staged behavior edits. Save or discard the batch below; Dashboard shows saved labels.</p>}
              {segments.length > 0 && (
                <>
                  <Ethogram
                    original={session.segments}
                    segments={segments}
                    duration={session.duration}
                    time={time}
                    range={range}
                    setRange={setRange}
                    seek={seek}
                    stale={stale.labels}
                    draft={labelDraft ? label : undefined}
                  />
                  {tab === 'review' && (
                    <LabelEditor
                      session={session}
                      edits={edits}
                      range={range}
                      setRange={setRange}
                      state={label}
                      setState={setLabel}
                      saveLabels={saveLabels}
                      resetLabels={resetLabels}
                      pending={pendingLabels}
                      stage={stageLabels}
                      saveStaged={() => void saveStagedLabels()}
                      removeStaged={(index) => setPendingLabels((pending) => pending.filter((_, i) => i !== index))}
                      discardStaged={() => setPendingLabels([])}
                    />
                  )}
                </>
              )}
              {tab === 'review' && hasTracking && <ReviewTools
                session={session}
                time={time}
                corrected={!!edits.scene || Object.keys(edits.frames).length > 0 || edits.labels.length > 0 || pendingLabels.length > 0}
                selectRange={(start, end) => {
                  videoRef.current?.pause()
                  const from = Math.max(0, Math.min(start, session.duration))
                  setRange([from, Math.max(from, Math.min(end, session.duration))])
                  seek(from)
                }}
              />}
              {session.measurements.length > 0 && (
                <Traces
                  session={session}
                  time={time}
                  seek={seek}
                  stale={stale.measurements}
                />
              )}
              {segments.length > 0 && (
                <Card
                  title="Behavior segments"
                  eyebrow="Timestamped review log"
                  accessory={
                    <span className="small muted">
                      {segments.length} segments
                    </span>
                  }
                >
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Time range</th>
                          <th>Behavior</th>
                          <th>Duration</th>
                          <th>Source</th>
                        </tr>
                      </thead>
                      <tbody>
                        {segments.map((s, i) => (
                          <tr key={i}>
                            <td>
                              <button
                                className="text-button"
                                onClick={() => seek(s.start_s)}
                              >
                                {s.start_s.toFixed(2)}–{s.end_s.toFixed(2)} s
                              </button>
                            </td>
                            <td>
                              <span className="table-state">
                                <i style={{ background: COLORS[s.state] }} />
                                {s.state}
                              </span>
                            </td>
                            <td>{(s.end_s - s.start_s).toFixed(2)} s</td>
                            <td>
                              <span
                                className={`badge ${s.source === 'manual' ? 'manual' : 'neutral'}`}
                              >
                                {s.source}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
            </div>
            <div className="workspace-side">
              {tab === 'dashboard' ? (
                <>
                  {current && (
                    <Card
                      title="Current behavior"
                      eyebrow="At the playhead"
                      accessory={
                        <span className="small muted">{time.toFixed(2)} s</span>
                      }
                    >
                      <div className="current-behavior">
                        <span
                          className="behavior-symbol"
                          style={{
                            background: `${COLORS[current.state]}18`,
                            color: COLORS[current.state],
                          }}
                        >
                          <Icon name="fish" size={28} />
                        </span>
                        <div>
                          <h3>{current.state}</h3>
                          <span
                            className={`badge ${current.source === 'manual' ? 'manual' : 'neutral'}`}
                          >
                            {current.source === 'manual'
                              ? 'Manual label'
                              : 'Automatic label'}
                          </span>
                        </div>
                      </div>
                      <p className="small muted">
                        {stale.labels
                          ? 'Automatic labels are stale after a scene/tracking correction.'
                          : 'Explore the ethogram to inspect behavior at any moment.'}
                      </p>
                      <button
                        className="text-button review-link"
                        onClick={() => changeTab('review')}
                      >
                        Review this frame
                        <Icon name="arrow" size={16} />
                      </button>
                    </Card>
                  )}
                  {results}
                  {segments.length > 0 && (
                    <StateSummary
                      segments={segments}
                      duration={session.duration}
                    />
                  )}
                  <Card title="Session details" eyebrow="Provenance">
                    <dl className="details-list">
                      <dt>Tracking format</dt>
                      <dd>prepds overlay fields</dd>
                      <dt>Prediction format</dt>
                      <dd>DCS compound output</dd>
                      <dt>Manual corrections</dt>
                      <dd>{editCount}</dd>
                      <dt>Persistence</dt>
                      <dd>Until page refresh</dd>
                    </dl>
                  </Card>
                </>
              ) : (
                <>
                  <Editors
                    session={session}
                    edits={edits}
                    reviewer={reviewer}
                    setReviewer={setReviewer}
                    tool={tool}
                    setTool={setTool}
                    keypoint={keypoint}
                    setKeypoint={setKeypoint}
                    scene={sceneDraft}
                    setScene={setSceneDraft}
                    frame={frameDraft}
                    setFrame={setFrameDraft}
                    frameIndex={frameIndex}
                    saveScene={() =>
                      apply((e) => {
                        validateScene(
                          sceneDraft,
                          session.overlay.width,
                          session.overlay.height,
                        )
                        return {
                          ...e,
                          scene: {
                            ...structuredClone(sceneDraft),
                            reviewer: reviewer.trim(),
                          },
                        }
                      }, 'Scene correction saved. Dependent results are stale.')
                    }
                    saveFrame={() =>
                      apply((e) => {
                        validateFrame(
                          frameDraft,
                          session.overlay.width,
                          session.overlay.height,
                        )
                        return {
                          ...e,
                          frames: {
                            ...e.frames,
                            [frameIndex]: {
                              ...structuredClone(frameDraft),
                              reviewer: reviewer.trim(),
                            },
                          },
                        }
                      }, `Frame ${frameIndex} correction saved. Dependent results are stale.`)
                    }
                    resetScene={async () => {
                      if (await apply((e) => ({ ...e, scene: null }), 'Automatic scene restored.')) setSceneDraft(initialScene())
                    }}
                    resetFrame={async () => {
                      if (await apply((e) => {
                        const frames = { ...e.frames }
                        delete frames[frameIndex]
                        return { ...e, frames }
                      }, 'Automatic frame restored.')) setFrameDraft(frameAt(session, frameIndex, emptyEdits()))
                    }}
                  />
                  <Card
                    title="Review decision"
                    eyebrow="Separate from the drug result"
                  >
                    <p className="small muted">
                      Accepting this review does not validate the model or clear
                      stale results.
                    </p>
                    {pendingLabels.length > 0 && <p className="small muted">Save or discard staged labels before deciding.</p>}
                    <div className="button-row">
                      <button
                        className="primary"
                        disabled={pendingLabels.length > 0}
                        onClick={() => void apply((e) => e, `Review accepted by ${reviewer.trim()}.`, 'ACCEPTED')}
                      >
                        Accept review
                      </button>
                      <button
                        disabled={pendingLabels.length > 0}
                        onClick={() => void apply((e) => e, `Review rejected by ${reviewer.trim()}.`, 'REJECTED')}
                      >
                        Reject review
                      </button>
                    </div>
                  </Card>
                  {results}
                  {segments.length > 0 && (
                    <StateSummary
                      segments={segments}
                      duration={session.duration}
                      detailed
                    />
                  )}
                </>
              )}
            </div>
          </div>
          <footer className="page-footer">
            <span>FishLab</span>
            <span>Zebrafish behavior analysis</span>
          </footer>
        </main>
      </div>
      <Chat ask={askChat} available={!!session.chat_available} open={chatOpen} onOpenChange={setChatOpen} />
    </fieldset>
  )
}
