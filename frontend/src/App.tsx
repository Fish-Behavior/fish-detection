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
  Point,
  Scene,
  SessionData,
} from './model.ts'
import { askChat, loadSession } from './api.ts'
import { Card, Icon } from './ui.tsx'
import VideoPanel from './VideoPanel.tsx'
import type { EditTool } from './VideoPanel.tsx'
import Editors, { LabelEditor } from './Editors.tsx'
import { Ethogram, StateSummary, Traces } from './Charts.tsx'
import Chat from './Chat.tsx'

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
  const [load, setLoad] = useState<
    | { status: 'loading' }
    | { status: 'error'; message: string }
    | { status: 'ready'; session: SessionData | null }
  >({ status: 'loading' })
  useEffect(() => {
    loadSession().then(
      (session) => setLoad({ status: 'ready', session }),
      (e) =>
        setLoad({
          status: 'error',
          message: e instanceof Error ? e.message : 'The request failed.',
        }),
    )
  }, [])
  if (load.status === 'loading') return <Empty title="Loading…" />
  if (load.status === 'error')
    return <Empty title="Could not load data" message={load.message} />
  if (!load.session)
    return (
      <Empty
        title="No data yet"
        message="Sessions appear here once the backend provides them."
      />
    )
  return <Workspace session={load.session} />
}

function Workspace({ session }: { session: SessionData }) {
  const initialScene = (): Scene => structuredClone(session.scene)
  const firstState: Behavior = session.segments[0]?.state ?? 'Undetermined'
  const firstRange = (): [number, number] => [0, Math.min(1, session.duration)]
  const [tab, setTab] = useState<'dashboard' | 'review'>('dashboard')
  const [chatOpen, setChatOpen] = useState(false)
  const [edits, setEdits] = useState<Edits>(emptyEdits),
    [reviewer, setReviewer] = useState('')
  const [time, setTime] = useState(0),
    [duration, setDuration] = useState(session.duration)
  const [tool, setTool] = useState<EditTool>('inspect'),
    [keypoint, setKeypoint] = useState('snout')
  const [sceneDraft, setSceneDraft] = useState(initialScene),
    [frameDraft, setFrameDraft] = useState(() =>
      frameAt(session, 0, emptyEdits()),
    )
  const [range, setRange] = useState<[number, number]>(firstRange),
    [label, setLabel] = useState<Behavior>(firstState)
  const [compound, setCompound] = useState(
      session.predictions?.predicted ?? '',
    ),
    [dose, setDose] = useState('')
  const [reviewStatus, setReviewStatus] = useState('PROCESSED_AUTO'),
    [filter, setFilter] = useState('all')
  const [notice, setNotice] = useState(''),
    [error, setError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const hasTracking = session.overlay.t.length > 0,
    frameIndex = Math.max(
      0,
      Math.min(
        Math.round(time * session.overlay.fps),
        session.overlay.t.length - 1,
      ),
    )
  const stale = staleFor(edits),
    segments = effectiveSegments(session.segments, edits.labels)
  const predictions = session.predictions
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
  }, [frameIndex, edits.frames])
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
  const apply = (action: () => void, message: string) => {
    try {
      validateReviewer(reviewer)
      action()
      setReviewStatus('EDITED')
      announce(message)
    } catch (e) {
      setNotice('')
      setError(
        e instanceof Error ? e.message : 'The correction could not be applied.',
      )
    }
  }
  const reset = () => {
    videoRef.current?.pause()
    setTime(0)
    setTool('inspect')
    setEdits(emptyEdits())
    setSceneDraft(initialScene())
    setFrameDraft(frameAt(session, 0, emptyEdits()))
    setRange(firstRange())
    setCompound(session.predictions?.predicted ?? '')
    setDose('')
    setReviewStatus('PROCESSED_AUTO')
    setFilter('all')
    announce(
      'Corrections reset to the automatic baseline. Chat has its own clear control.',
    )
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
        waterline: clamp(s.waterline + dy, height),
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
    apply(() => {
      validateInterval(range[0], range[1], session.duration)
      setEdits((e) => ({
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
      }))
    }, 'Behavior correction saved. Timeline and totals updated; feature/prediction results are stale.')
  const resetLabels = () => {
    setEdits((e) => ({ ...e, labels: [] }))
    setReviewStatus('EDITED')
    announce('Automatic behavior labels restored.')
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
                apply(() => {
                  if (!compound.trim())
                    throw new Error('Enter a final compound.')
                  setEdits((e) => ({
                    ...e,
                    finalResult: {
                      compound: compound.trim(),
                      dose: dose.trim(),
                      reviewer: reviewer.trim(),
                    },
                  }))
                }, 'Manual final result saved. Automatic probabilities are unchanged.')
              }
            >
              Apply final result
            </button>
            <button
              onClick={() => {
                setEdits((e) => ({ ...e, finalResult: null }))
                setCompound(session.predictions?.predicted ?? '')
                setDose('')
                setReviewStatus('EDITED')
                announce('Manual final result removed.')
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
        onClick={() =>
          announce(
            'Backend required: no real tracking, feature extraction, or inference ran. Stale flags remain until real recalculation is connected.',
          )
        }
      >
        <Icon name="reset" />
        Rerun affected stages
      </button>
    </Card>
  )

  return (
    <div className="app-shell">
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
            <span>01</span>
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
          {filter === 'all' ||
          (filter === 'edited' && editCount > 0) ||
          (filter === 'needs-review' &&
            !['ACCEPTED', 'REJECTED'].includes(reviewStatus)) ? (
            <div className="session-item selected">
              <span className="session-dot" />
              <span>
                <b>{session.video_id}</b>
                <small>{session.name}</small>
              </span>
            </div>
          ) : (
            <p className="small muted">No matching sessions.</p>
          )}
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
                  {session.name} · {session.duration} s
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
                    />
                  )}
                </>
              )}
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
                      apply(() => {
                        validateScene(
                          sceneDraft,
                          session.overlay.width,
                          session.overlay.height,
                        )
                        setEdits((e) => ({
                          ...e,
                          scene: {
                            ...structuredClone(sceneDraft),
                            reviewer: reviewer.trim(),
                          },
                        }))
                      }, 'Scene correction saved. Dependent results are stale.')
                    }
                    saveFrame={() =>
                      apply(() => {
                        validateFrame(
                          frameDraft,
                          session.overlay.width,
                          session.overlay.height,
                        )
                        setEdits((e) => ({
                          ...e,
                          frames: {
                            ...e.frames,
                            [frameIndex]: {
                              ...structuredClone(frameDraft),
                              reviewer: reviewer.trim(),
                            },
                          },
                        }))
                      }, `Frame ${frameIndex} correction saved. Dependent results are stale.`)
                    }
                    resetScene={() => {
                      setEdits((e) => ({ ...e, scene: null }))
                      setSceneDraft(initialScene())
                      setReviewStatus('EDITED')
                      announce(
                        'Automatic scene restored. Other corrections remain.',
                      )
                    }}
                    resetFrame={() => {
                      setEdits((e) => {
                        const frames = { ...e.frames }
                        delete frames[frameIndex]
                        return { ...e, frames }
                      })
                      setFrameDraft(frameAt(session, frameIndex, emptyEdits()))
                      setReviewStatus('EDITED')
                      announce('Automatic frame restored.')
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
                    <div className="button-row">
                      <button
                        className="primary"
                        onClick={() => {
                          try {
                            validateReviewer(reviewer)
                            setReviewStatus('ACCEPTED')
                            announce(
                              `Review accepted by ${reviewer.trim()}. Results and stale flags are unchanged.`,
                            )
                          } catch (e) {
                            setError((e as Error).message)
                          }
                        }}
                      >
                        Accept review
                      </button>
                      <button
                        onClick={() => {
                          try {
                            validateReviewer(reviewer)
                            setReviewStatus('REJECTED')
                            announce(`Review rejected by ${reviewer.trim()}.`)
                          } catch (e) {
                            setError((e as Error).message)
                          }
                        }}
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
      {askChat && (
        <Chat ask={askChat} open={chatOpen} onOpenChange={setChatOpen} />
      )}
    </div>
  )
}
