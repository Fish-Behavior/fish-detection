import { useCallback, useEffect, useRef, useState } from 'react'
import {
  COLORS,
  demoProvider,
  effectiveSegments,
  emptyEdits,
  frameAt,
  staleFor,
  validateFrame,
  validateInterval,
  validateReviewer,
  validateScene,
} from './model.ts'
import type { Behavior, Box, Edits, Point, Scene } from './model.ts'
import { Card, Icon } from './ui.tsx'
import VideoPanel from './VideoPanel.tsx'
import type { EditTool } from './VideoPanel.tsx'
import Editors from './Editors.tsx'
import { Ethogram, StateSummary, Traces } from './Charts.tsx'
import Chat from './Chat.tsx'

const session = demoProvider.session()
const initialScene = (): Scene => structuredClone(session.scene)

export default function App() {
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
  const [range, setRange] = useState<[number, number]>([2.5, 4]),
    [label, setLabel] = useState<Behavior>('Controlled Swim')
  const [compound, setCompound] = useState('COMPOUND_A'),
    [dose, setDose] = useState('')
  const [reviewStatus, setReviewStatus] = useState('PROCESSED_AUTO'),
    [filter, setFilter] = useState('all')
  const [notice, setNotice] = useState(''),
    [error, setError] = useState('')
  const [local, setLocal] = useState<{ url: string; name: string } | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null),
    fileRef = useRef<HTMLInputElement>(null)
  const localRef = useRef<string | null>(null)
  const isLocal = !!local,
    frameIndex = Math.min(
      Math.round(time * session.overlay.fps),
      session.overlay.t.length - 1,
    )
  const stale = staleFor(edits),
    segments = effectiveSegments(session.segments, edits.labels)
  const predictedProbability =
    session.predictions[`p:${session.predictions.predicted}`]
  const current =
    segments.find((s) => s.start_s <= time && s.end_s > time) ??
    segments.at(-1)!
  const editCount =
    Object.keys(edits.frames).length +
    edits.labels.length +
    Number(!!edits.scene) +
    Number(!!edits.finalResult)
  useEffect(() => {
    setFrameDraft(frameAt(session, frameIndex, edits))
  }, [frameIndex, edits.frames])
  useEffect(
    () => () => {
      if (localRef.current) URL.revokeObjectURL(localRef.current)
    },
    [],
  )
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
  const selectDemo = () => {
    videoRef.current?.pause()
    if (localRef.current) URL.revokeObjectURL(localRef.current)
    localRef.current = null
    setLocal(null)
    setTime(0)
    setDuration(session.duration)
    setTool('inspect')
    announce('Synthetic demo selected.')
  }
  const chooseFile = (file: File) => {
    videoRef.current?.pause()
    if (localRef.current) URL.revokeObjectURL(localRef.current)
    localRef.current = URL.createObjectURL(file)
    setLocal({ url: localRef.current, name: file.name })
    setTime(0)
    setDuration(0)
    setTool('inspect')
    announce('Local playback only. This file is not uploaded or analyzed.')
  }
  const reset = () => {
    selectDemo()
    setEdits(emptyEdits())
    setSceneDraft(initialScene())
    setFrameDraft(frameAt(session, 0, emptyEdits()))
    setRange([2.5, 4])
    setCompound('COMPOUND_A')
    setDose('')
    setReviewStatus('PROCESSED_AUTO')
    setFilter('all')
    announce(
      'Demo corrections restored to the automatic baseline. Chat has its own clear control.',
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
      setFrameDraft((f) => ({
        ...f,
        keypoints: {
          ...f.keypoints,
          [keypoint]: [p[0], p[1], f.keypoints[keypoint][2]],
        },
      }))
  }
  const changeTab = (next: 'dashboard' | 'review') => {
    setTab(next)
    setTool('inspect')
  }
  const results = (
    <Card
      title="Compound prediction"
      eyebrow="Illustrative drug results"
      accessory={
        <span className={`badge ${stale.predictions ? 'warning' : 'neutral'}`}>
          {stale.predictions ? 'Stale' : 'Sample data'}
        </span>
      }
    >
      <div className="prediction-lead">
        <span className="muted small">Automatic model result</span>
        <div>
          <strong>{session.predictions.predicted.replaceAll('_', ' ')}</strong>
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
        {Object.entries(session.predictions)
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
                    background: ['#2c70b6', '#8ba8c8', '#c7d6e4'][i],
                  }}
                />
              </div>
            </div>
          ))}
      </div>
      <p className="result-note">
        <Icon name="info" size={15} />
        These numbers illustrate the UI; they are not model evidence.
      </p>
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
                setCompound('COMPOUND_A')
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
            Analysis dashboard
          </button>
          <button
            className={tab === 'review' ? 'active' : ''}
            onClick={() => changeTab('review')}
          >
            <Icon name="review" />
            Review & corrections
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
            <button
              className={`session-item ${!isLocal ? 'selected' : ''}`}
              onClick={selectDemo}
            >
              <span className="session-dot" />
              <span>
                <b>{session.video_id}</b>
                <small>Single fish · synthetic</small>
              </span>
              <Icon name="arrow" size={14} />
            </button>
          ) : (
            <p className="small muted">No matching demo sessions.</p>
          )}
          {local && (
            <div className="local-session">
              <b>{local.name}</b>
              <small>Local playback · no analysis</small>
            </div>
          )}
          <button
            className="choose-file"
            onClick={() => fileRef.current?.click()}
          >
            <Icon name="upload" />
            Choose local video
          </button>
          <input
            ref={fileRef}
            hidden
            aria-label="Choose local video file"
            type="file"
            accept="video/*,.mp4,.webm,.mov"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) chooseFile(file)
              e.target.value = ''
            }}
          />
        </div>
        <div className="sidebar-footer">
          <span className="live-dot" />
          <div>
            <b>Demo workspace</b>
            <small>In this tab · no backend</small>
          </div>
          <span className="version">v0.1</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            <b>Workspace</b>
            <span className="crumb">/</span>
            {tab === 'dashboard' ? 'Analysis' : 'Review'}
          </span>
          <span className="preview-pill">
            <span />
            Frontend preview
          </span>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <p className="eyebrow">ZEBRAFISH · SESSION WORKSPACE</p>
              <h1>
                {tab === 'dashboard'
                  ? 'From movement to insight.'
                  : 'A closer look. A clearer result.'}
              </h1>
              <p className="subtitle">
                {tab === 'dashboard'
                  ? 'Follow the pipeline, explore behavior, and review every result.'
                  : 'Inspect the evidence and keep your corrections alongside the original.'}
              </p>
            </div>
            <button className="reset-button" onClick={reset}>
              <Icon name="reset" />
              Reset demo
            </button>
          </div>
          <div className="demo-banner">
            <Icon name="info" />
            <p>
              <b>Synthetic demonstration.</b> Video, overlays, and probabilities
              are illustrative. Edits and chat reset on refresh.
            </p>
            <span className="badge neutral">Browser only</span>
          </div>
          <div className="session-heading">
            <div className="session-heading-main">
              <span className="session-icon">
                <Icon name="fish" size={23} />
              </span>
              <div>
                <h2>{local?.name ?? session.video_id}</h2>
                <p>
                  {isLocal
                    ? 'Local file · analysis unavailable'
                    : 'Synthetic tank · single fish · 12 seconds'}
                </p>
              </div>
            </div>
            <span
              className={`badge ${reviewStatus === 'ACCEPTED' ? 'success' : 'neutral'}`}
            >
              {isLocal ? 'NOT PROCESSED' : reviewStatus.replaceAll('_', ' ')}
            </span>
          </div>
          <ol className="pipeline" aria-label="Pipeline stages">
            {['Scene', 'Tracking', 'Features', 'Behavior', 'Prediction'].map(
              (stage, i) => {
                const s =
                  !isLocal &&
                  (i === 2
                    ? stale.measurements
                    : i === 3
                      ? stale.labels
                      : i === 4
                        ? stale.predictions
                        : false)
                return (
                  <li key={stage} className={s ? 'stale' : ''}>
                    <span className="stage-number">
                      {!isLocal && !s ? (
                        <Icon name="check" size={14} />
                      ) : (
                        `0${i + 1}`
                      )}
                    </span>
                    <div>
                      <b>{stage}</b>
                      <small>
                        {isLocal
                          ? 'Not processed'
                          : s
                            ? 'Needs rerun'
                            : [
                                'Scene defined',
                                'Sample track',
                                'Sample metrics',
                                'Sample labels',
                                'Sample output',
                              ][i]}
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
                src={local?.url ?? session.video_url}
                isLocal={isLocal}
                title={local?.name ?? session.name}
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
              />
              {!isLocal && (
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
                  />
                  <Traces
                    session={session}
                    time={time}
                    seek={seek}
                    stale={stale.measurements}
                  />
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
                </>
              )}
            </div>
            <div className="workspace-side">
              {isLocal ? (
                <Card title="Ready for backend processing" eyebrow="Local file">
                  <p className="muted">
                    Your video is playing locally. Synthetic charts and
                    predictions are hidden because they do not describe this
                    file.
                  </p>
                  <button className="primary" onClick={selectDemo}>
                    Return to synthetic demo
                  </button>
                </Card>
              ) : tab === 'dashboard' ? (
                <>
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
                            : 'Sample automatic label'}
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
                  {results}
                  <StateSummary
                    segments={segments}
                    duration={session.duration}
                  />
                  <Card title="Session details" eyebrow="Provenance">
                    <dl className="details-list">
                      <dt>Source</dt>
                      <dd>Generated illustration</dd>
                      <dt>Tracking format</dt>
                      <dd>prepds overlay fields</dd>
                      <dt>Prediction format</dt>
                      <dd>DCS compound output</dd>
                      <dt>Model verdict</dt>
                      <dd>Illustrative, not evaluated</dd>
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
                    range={range}
                    setRange={setRange}
                    state={label}
                    setState={setLabel}
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
                    saveLabels={() =>
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
                    resetLabels={() => {
                      setEdits((e) => ({ ...e, labels: [] }))
                      setReviewStatus('EDITED')
                      announce('Automatic behavior labels restored.')
                    }}
                  />
                  {results}
                  <StateSummary
                    segments={segments}
                    duration={session.duration}
                  />
                  <Card
                    title="Review decision"
                    eyebrow="Separate from the drug result"
                  >
                    <p className="small muted">
                      Accepting this demo review does not validate the model or
                      clear stale results.
                    </p>
                    <div className="button-row">
                      <button
                        className="primary"
                        onClick={() => {
                          try {
                            validateReviewer(reviewer)
                            setReviewStatus('ACCEPTED')
                            announce(
                              `Demo review accepted by ${reviewer.trim()}. Results and stale flags are unchanged.`,
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
                            announce(
                              `Demo review rejected by ${reviewer.trim()}.`,
                            )
                          } catch (e) {
                            setError((e as Error).message)
                          }
                        }}
                      >
                        Reject review
                      </button>
                    </div>
                  </Card>
                </>
              )}
            </div>
          </div>
          <footer className="page-footer">
            <span>FishLab · Frontend scaffold</span>
            <span>React preview · synthetic data · no API calls</span>
          </footer>
        </main>
      </div>
      <Chat open={chatOpen} onOpenChange={setChatOpen} />
    </div>
  )
}
