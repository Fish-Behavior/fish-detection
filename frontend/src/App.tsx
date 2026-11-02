import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import type { SetStateAction } from 'react'
import {
  COLORS,
  effectiveSegments,
  emptyEdits,
  frameAt,
  frameIndexAt,
  hasEdits,
  draftJSON,
  parseDraft,
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
  FrameValue,
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
import utLogo from './assets/ut-logo.png'

// Plain words for the stored review and processing codes.
const REVIEW_LABELS: Record<string, string> = {
  PROCESSED_AUTO: 'Not reviewed yet', EDITED: 'Has corrections', ACCEPTED: 'Review accepted',
  REJECTED: 'Review rejected', NOT_PROCESSED: 'Not analyzed yet',
}
const PROCESSING_LABELS: Record<string, string> = {
  processed: 'Analyzed', queued: 'Waiting to analyze', running: 'Analyzing', failed: 'Analysis failed',
  interrupted: 'Analysis stopped', idle: 'Not analyzed yet', unprocessed: 'Not analyzed yet',
}
const plain = (labels: Record<string, string>, code: string) => labels[code] ?? code.replaceAll('_', ' ').toLowerCase()
// 1200.9 -> "20 min 1 s"
const minutes = (seconds: number) => seconds >= 60 ? `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s` : `${seconds.toFixed(1)} s`

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
    if (pendingCount.current && !window.confirm('Autosave is unavailable in this browser. Discard unsaved review edits and reload?')) return
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
    <header className={`brand-header${load.status === 'ready' && load.session ? ' main-shell' : ''}`}>
      <img src={utLogo} alt="The University of Toledo" width="723" height="276" />
      <button onClick={() => void reload()} title="Load the latest recordings and results">
        <Icon name="reset" />
        Refresh
      </button>
    </header>
    {load.status === 'loading' ? <Empty title="Loading…" /> : load.status === 'error'
      ? <Empty title={load.stage === 'connection' ? 'Could not connect to the analysis server' : load.stage === 'inventory' ? 'Could not load the recording list' : 'Could not load this recording'} message={load.message} /> : !load.session
      ? <Empty title="No recordings yet" message="Add recordings to the configured video folder, then press Refresh." />
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
  const draftKey = `fishlab:review-draft:${session.video_id}`
  // Unsaved work from an earlier visit; read once, before the drafts below take their first values.
  const [restored] = useState(() => {
    try { return parseDraft(localStorage.getItem(draftKey), session.baseline ?? null) } catch { return null }
  })
  const [edits, setEdits] = useState<Edits>(() => structuredClone(session.review?.edits ?? emptyEdits())),
    [reviewer, setReviewer] = useState(restored?.reviewer || session.review?.reviewer || '')
  const [time, setTime] = useState(0),
    [duration, setDuration] = useState(session.duration)
  const [tool, setTool] = useState<EditTool>('inspect'),
    [keypoint, setKeypoint] = useState('snout')
  const [sceneDraft, setSceneDraft] = useState(() => structuredClone(restored?.scene ?? edits.scene ?? session.scene))
  // Frame corrections stay staged across frame changes until saved or discarded.
  const [pendingFrames, setPendingFrames] = useState<Record<number, FrameValue>>(() => restored?.frames ?? {})
  const [range, setRange] = useState<[number, number]>(firstRange),
    [label, setLabel] = useState<Behavior>(firstState)
  const [pendingLabels, setPendingLabels] = useState<LabelEdit[]>(() => restored?.labels ?? [])
  const [autosaved, setAutosaved] = useState<Date | 'failed' | null>(null)
  const [compound, setCompound] = useState(
      edits.finalResult?.compound ?? session.predictions?.predicted ?? '',
    ),
    [dose, setDose] = useState(edits.finalResult?.dose ?? '')
  const [reviewStatus, setReviewStatus] = useState(review.decision ?? (hasEdits(edits) ? 'EDITED' : session.has_tracking ? 'PROCESSED_AUTO' : 'NOT_PROCESSED')),
    [filter, setFilter] = useState('all')
  const [notice, setNotice] = useState(restored ? `Restored unsaved review edits autosaved at ${new Date(restored.saved_at).toLocaleString()}.${restored.revision !== review.revision ? ' The review was saved elsewhere since then: check the staged frames and scene before saving, as they replace the newer values.' : ''} Save all to keep them on the backend.` : ''),
    [error, setError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const hasTracking = session.has_tracking ?? session.overlay.t.length > 0,
    frameIndex = frameIndexAt(session, time)
  const stale = session.stale ?? staleFor(edits)
  const savedSegments = session.reviewed_segments ?? effectiveSegments(session.segments, edits.labels)
  const segments = tab === 'review' ? effectiveSegments(savedSegments, pendingLabels) : savedSegments
  const frameDraft = pendingFrames[frameIndex] ?? frameAt(session, frameIndex, edits)
  // Edits this frame's staged draft; a draft edited back to the saved value is unstaged.
  // Tracking for a frame outside the loaded overlay window reads as empty; a draft built on it would erase the box and keypoints.
  const frameLoaded = session.overlay.frame_idx ? session.overlay.frame_idx.includes(frameIndex) : frameIndex < session.overlay.t.length
  const setFrameDraft = (action: SetStateAction<FrameValue>) => {
    if (!pendingFrames[frameIndex] && !frameLoaded) { setError('Tracking for this frame is still loading. Wait a moment, then correct it.'); return }
    setPendingFrames((pending) => {
      const saved = frameAt(session, frameIndex, edits)
      const next = typeof action === 'function' ? action(pending[frameIndex] ?? saved) : action
      const rest = { ...pending }
      delete rest[frameIndex]
      // While a save is in flight the saved value is about to change, so nothing unstages by equality.
      return JSON.stringify(next) === JSON.stringify(saved) && !busyRef.current ? rest : { ...rest, [frameIndex]: next }
    })
  }
  const savedScene = edits.scene ?? session.scene
  const sceneDirty = sceneDraft.waterline !== savedScene.waterline || sceneDraft.roi.some((v, i) => v !== savedScene.roi[i])
  const stagedFrames = Object.keys(pendingFrames).length
  const pendingTotal = stagedFrames + pendingLabels.length + Number(sceneDirty)
  // Only the tab that wrote the draft removes it, so a second tab on the same video never deletes another's work.
  // ponytail: two tabs both editing one video still share the key (last write wins); a per-tab key if that matters.
  const tabId = useRef(Math.random().toString(36).slice(2))
  const ownsDraft = useRef(!!restored)
  // Remove the stored draft only if this tab wrote the copy that is there now.
  const dropOwnDraft = () => {
    if (ownsDraft.current && parseDraft(localStorage.getItem(draftKey), session.baseline ?? null)?.tab === tabId.current) localStorage.removeItem(draftKey)
  }
  useEffect(() => {
    try {
      if (!pendingTotal) {
        dropOwnDraft()
        ownsDraft.current = false
        setAutosaved(null)
        return
      }
      localStorage.setItem(draftKey, draftJSON({
        tab: tabId.current, baseline: session.baseline ?? null, revision: review.revision, saved_at: new Date().toISOString(), reviewer,
        scene: sceneDirty ? { roi: sceneDraft.roi, waterline: sceneDraft.waterline } : null,
        frames: pendingFrames, labels: pendingLabels,
      }))
      ownsDraft.current = true
      setAutosaved(new Date())
    } catch {
      // An older, partial draft must not come back later as if it were complete.
      try { dropOwnDraft() } catch { /* storage unavailable */ }
      setAutosaved('failed')
    }
  }, [draftKey, session.baseline, review.revision, reviewer, pendingTotal, sceneDirty, sceneDraft, pendingFrames, pendingLabels])
  // Without browser storage, unsaved work is only in memory: warn before it is lost.
  const unprotected = autosaved === 'failed' ? pendingTotal : 0
  useEffect(() => {
    onPendingChange(unprotected)
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    if (unprotected) window.addEventListener('beforeunload', warn)
    return () => { onPendingChange(0); window.removeEventListener('beforeunload', warn) }
  }, [unprotected, onPendingChange])
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
  // `onSaved` runs as soon as the backend accepts the save, before the slower session refresh.
  const apply = async (action: (edits: Edits) => Edits, message: string, decision: ReviewData['decision'] = null, onSaved?: () => void) => {
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const next = action(structuredClone(edits))
      const saved = await saveReview(session, review, next, reviewer.trim(), decision)
      setReview(saved); setEdits(saved.edits); onSaved?.()
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
    if (!window.confirm('Reset every saved and unsaved correction for this recording to the automatic baseline?')) return
    if (!await apply(() => emptyEdits(), 'Corrections reset to the automatic baseline.')) return
    setPendingLabels([])
    setPendingFrames({})
    videoRef.current?.pause()
    setTime(0)
    setTool('inspect')
    setSceneDraft(initialScene())
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
    }, 'Behavior correction saved. Update the results to include it.')
  const stageLabels = () => {
    try {
      validateReviewer(reviewer)
      validateInterval(range[0], range[1], session.duration)
      setPendingLabels((pending) => [...pending, { start_s: range[0], end_s: range[1], state: label, reviewer: reviewer.trim() }])
      announce('Behavior edit staged. Save the batch to persist it.')
    } catch (e) { setError((e as Error).message) }
  }
  // After a save, unstage only what was sent: edits made while it was in flight stay staged.
  const unstage = (frames: Record<number, FrameValue>, labels: LabelEdit[]) => {
    setPendingFrames((pending) => Object.fromEntries(Object.entries(pending).filter(([k, f]) => frames[Number(k)] !== f)))
    setPendingLabels((pending) => pending.filter((l) => !labels.includes(l)))
  }
  const saveStagedLabels = async () => {
    const sent = pendingLabels
    await apply((e) => ({ ...e, labels: [...e.labels, ...sent] }), 'Behavior corrections saved. Update the results to include them.', null, () => unstage({}, sent))
  }
  // One backend save for everything staged: the scene draft, every staged frame and every staged label.
  const saveAll = async () => {
    const { width, height } = session.overlay
    const sentFrames = pendingFrames, sentLabels = pendingLabels
    await apply((e) => {
      if (sceneDirty) validateScene(sceneDraft, width, height)
      const frames = { ...e.frames }
      for (const [index, frame] of Object.entries(pendingFrames)) {
        try { validateFrame(frame, width, height) } catch (err) { throw new Error(`Frame ${index}: ${(err as Error).message}`) }
        frames[Number(index)] = { ...structuredClone(frame), reviewer: reviewer.trim() }
      }
      return {
        ...e,
        scene: sceneDirty ? { roi: [...sceneDraft.roi], waterline: sceneDraft.waterline, reviewer: reviewer.trim() } : e.scene,
        frames,
        labels: [...e.labels, ...pendingLabels],
      }
    }, 'All corrections saved. Update the results to include them.', null, () => unstage(sentFrames, sentLabels))
  }
  const discardAll = () => {
    if (!window.confirm(`Discard ${pendingTotal} unsaved review edit${pendingTotal === 1 ? '' : 's'}? This cannot be undone.`)) return
    setPendingFrames({}); setPendingLabels([]); setSceneDraft(structuredClone(savedScene))
    announce('Unsaved review edits discarded.')
  }
  const resetLabels = async () => {
    if (await apply((e) => ({ ...e, labels: [] }), 'Automatic behavior labels restored.')) setPendingLabels([])
  }
  const recalculate = async () => {
    if (pendingTotal) { setError('Save or discard your unsaved corrections first.'); return }
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const updated = await rerun(session, review, reviewer.trim())
      setSession({ ...updated, overlay: session.overlay }); setReview(updated.review!); setEdits(updated.review!.edits)
      announce('Measurements and behaviors updated. Rerun the drug prediction to update it too.')
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
    if (pendingTotal) { setError('Save or discard your unsaved corrections first.'); return }
    if (busyRef.current) return
    busyRef.current = true; setBusy(true)
    try {
      validateReviewer(reviewer)
      const updated = await predict(session, review, reviewer.trim())
      setSession({ ...updated, overlay: session.overlay }); setReview(updated.review!)
      announce('Drug prediction updated.')
    } catch (e) { setNotice(''); setError((e as Error).message) }
    finally { busyRef.current = false; setBusy(false) }
  }
  const changeTab = (next: 'dashboard' | 'review') => {
    setTab(next)
    setTool('inspect')
  }
  const results = (
    <Card
      title="Drug prediction"
      accessory={
        stale.predictions && <span className="badge warning">Out of date</span>
      }
    >
      {predictions ? (
        <>
          <div className="prediction-lead">
            <span className="muted small">Most likely compound</span>
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
          {/* The top three answer the question; the rest stay one click away. */}
          <div className="probabilities">
            {Object.entries(predictions!)
              .filter(([k]) => k.startsWith('p:'))
              .sort(([, a], [, b]) => Number(b) - Number(a))
              .slice(0, 3)
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
                        background: ['var(--blue)', 'var(--line-strong)', 'var(--blue-line)'][i],
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>
          <details className="all-probabilities">
            <summary>All compounds</summary>
            <table>
              <tbody>
                {Object.entries(predictions!)
                  .filter(([k]) => k.startsWith('p:'))
                  .sort(([, a], [, b]) => Number(b) - Number(a))
                  .map(([key, value]) => (
                    <tr key={key}><td>{key.slice(2).replaceAll('_', ' ')}</td><td>{(Number(value) * 100).toFixed(0)}%</td></tr>
                  ))}
              </tbody>
            </table>
          </details>
        </>
      ) : (
        <p className="muted small">
          No drug prediction for this recording yet.
        </p>
      )}
      <div className="dose-status">
        <span>Dose</span>
        <b>Not estimated by this model</b>
      </div>
      {edits.finalResult && (
        <div className="manual-result">
          <span className="badge manual">Manual final result</span>
          <h3>{edits.finalResult.compound}</h3>
          <p>
            {edits.finalResult.dose || 'Dose not specified'}, set by{' '}
            {edits.finalResult.reviewer}
          </p>
        </div>
      )}
      {tab === 'review' && <>
      <details className="result-editor">
        <summary>Set the final result manually</summary>
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
                }, 'Final result saved. The model percentages are unchanged.')
              }
            >
              Apply final result
            </button>
            <button
              onClick={async () => {
                if (await apply((e) => ({ ...e, finalResult: null }), 'Final result removed; showing the model result again.')) {
                  setCompound(session.predictions?.predicted ?? ''); setDose('')
                }
              }}
            >
              Restore model result
            </button>
          </div>
          <p className="small muted">
            A manual result does not change the model's percentages.
          </p>
        </div>
      </details>
      <button
        className="rerun-button"
        onClick={() => void recalculate()}
      >
        <Icon name="reset" />
        Update measurements and behaviors
      </button>
      <button className="rerun-button" onClick={() => void predictCompound()}>Rerun drug prediction</button>
      </>}
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
            FishLab<small>Zebrafish behavior</small>
          </span>
        </a>
        <nav aria-label="Views">
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
            <span>Recordings</span>
            <span>{sessions.length}</span>
          </div>
          <label className="sr-only" htmlFor="session-filter">
            Filter recordings
          </label>
          <select
            id="session-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All recordings</option>
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
                  <span className="session-status">{plain(PROCESSING_LABELS, item.processing_status)}{item.edited ? ', has corrections' : ''}</span>
                </span>
              </button>
            ))}
            {visibleSessions.length === 0 && <p className="small muted">No recordings match this filter.</p>}
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <main>
          <div className="page-heading">
            <div>
              <h1>{tab === 'dashboard' ? 'Dashboard' : 'Review'}</h1>
              <p className="subtitle">
                {tab === 'dashboard'
                  ? 'Watch the recording, see how the fish behaved, and check the drug prediction.'
                  : 'Correct the tracking and behavior labels. The original results are kept.'}
              </p>
            </div>
            {tab === 'review' && <button className="reset-button" onClick={reset}>
              <Icon name="reset" />
              Undo all corrections
            </button>}
          </div>
          <div className="session-heading">
            <div className="session-heading-main">
              <span className="session-icon">
                <Icon name="fish" size={23} />
              </span>
              <div>
                <h2>{session.video_id}</h2>
                <p>
                  {session.name}, {minutes(session.duration)}
                </p>
              </div>
            </div>
            <span
              className={`badge ${reviewStatus === 'ACCEPTED' ? 'success' : 'neutral'}`}
            >
              {plain(REVIEW_LABELS, reviewStatus)}
            </span>
          </div>
          <ol className="pipeline" aria-label="Pipeline stages">
            {['Setup', 'Fish tracking', 'Measurements', 'Behaviors', 'Drug prediction'].map(
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
                          ? 'Out of date'
                          : done
                            ? 'Done'
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
            {busy && <p className="notice">Saving…</p>}
            {session.warnings?.map((w) => <p key={w} className="inline-error">{w}</p>)}
            {processing && processing.status !== 'idle' && <p className="notice">{plain(PROCESSING_LABELS, processing.status)}{processing.message && `: ${processing.message}`}</p>}
            {!hasTracking && <div className="processing-controls">
              <label>Reviewer <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} maxLength={100} /></label>
              <button onClick={() => void processVideo()} disabled={!session.processable}>Run analysis</button>
              {!session.processable && <p className="muted">This video is not linked to a trial in the experiment workbook yet, so it cannot be analyzed.</p>}
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
              {tab === 'review' && (
                <section className={`draft-bar${pendingTotal ? ' unsaved' : ''}`} aria-label="Unsaved review edits">
                  <div>
                    <b>
                      {pendingTotal
                        ? `${[
                            stagedFrames && `${stagedFrames} frame${stagedFrames === 1 ? '' : 's'}`,
                            sceneDirty && 'scene',
                            pendingLabels.length && `${pendingLabels.length} label${pendingLabels.length === 1 ? '' : 's'}`,
                          ].filter(Boolean).join(', ')} not saved yet`
                        : 'No unsaved edits'}
                    </b>
                    <span className={autosaved === 'failed' ? 'draft-warning' : 'small muted'} role={autosaved === 'failed' ? 'alert' : undefined}>
                      {autosaved === 'failed'
                        ? 'Autosave is unavailable in this browser. Save before leaving.'
                        : autosaved
                          ? `Autosaved in this browser at ${autosaved.toLocaleTimeString()}.`
                          : 'Corrections autosave in this browser as you work.'}
                    </span>
                  </div>
                  <div className="button-row">
                    <button className="primary" disabled={!pendingTotal} onClick={() => void saveAll()}>Save all</button>
                    <button disabled={!pendingTotal} onClick={discardAll}>Discard</button>
                  </div>
                </section>
              )}
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
                setTool={setTool}
                keypoint={keypoint}
                sceneDraft={sceneDraft}
                frameDraft={frameDraft}
                sketch={sketch}
                nudge={nudge}
              />
              {tab === 'review' && pendingLabels.length > 0 && <p className="notice" role="status">Showing {pendingLabels.length} unsaved behavior correction{pendingLabels.length === 1 ? '' : 's'} on the timeline. The Dashboard shows saved labels only.</p>}
              {segments.length > 0 && (
                <>
                  <Ethogram
                    original={session.segments}
                    segments={segments}
                    duration={session.duration}
                    editing={tab === 'review'}
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
                corrected={!!edits.scene || Object.keys(edits.frames).length > 0 || edits.labels.length > 0 || pendingTotal > 0}
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
                  title="Behavior log"
                  accessory={
                    <span className="small muted">
                      {segments.length} entries
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
                          <th>Labeled by</th>
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
                                {s.source === 'manual' ? 'Reviewer' : 'Automatic'}
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
                      title="Behavior now"
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
                              ? 'Set by a reviewer'
                              : 'Automatic'}
                          </span>
                        </div>
                      </div>
                      <p className="small muted">
                        {stale.labels
                          ? 'Out of date after a tracking correction. Update the results on the Review page.'
                          : 'Click the timeline to jump to any moment.'}
                      </p>
                      <button
                        className="text-button review-link"
                        onClick={() => changeTab('review')}
                      >
                        Correct this moment
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
                  <Card title="Recording details">
                    <dl className="details-list">
                      <dt>Length</dt>
                      <dd>{minutes(session.duration)}</dd>
                      <dt>Frame rate</dt>
                      <dd>{Math.round(session.overlay.fps)} frames per second</dd>
                      <dt>Picture size</dt>
                      <dd>{session.overlay.width} × {session.overlay.height} pixels</dd>
                      <dt>Saved corrections</dt>
                      <dd>{editCount}</dd>
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
                      }, 'Scene correction saved. Update the results to include it.')
                    }
                    saveFrame={async () => {
                      const index = frameIndex, sent = pendingFrames[index]
                      // 1: the same guard as drawing, so an unloaded frame is never saved as blank.
                      if (!sent && !frameLoaded) { setError('Tracking for this frame is still loading. Wait a moment, then apply it.'); return }
                      await apply((e) => {
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
                      }, `Frame ${frameIndex} correction saved. Update the results to include it.`, null, () => unstage(sent ? { [index]: sent } : {}, []))
                    }}
                    resetScene={async () => {
                      if (await apply((e) => ({ ...e, scene: null }), 'Automatic scene restored.')) setSceneDraft(initialScene())
                    }}
                    resetFrame={async () => {
                      const index = frameIndex
                      if (await apply((e) => {
                        const frames = { ...e.frames }
                        delete frames[index]
                        return { ...e, frames }
                      }, 'Automatic frame restored.')) setPendingFrames(({ [index]: _restored, ...rest }) => rest)
                    }}
                  />
                  <Card
                    title="Review decision"
                  >
                    <p className="small muted">
                      Accepting confirms the tracking and labels. It does not
                      check the drug prediction or update out-of-date results.
                    </p>
                    {pendingTotal > 0 && <p className="small muted">Save or discard your unsaved corrections first.</p>}
                    <div className="button-row">
                      <button
                        className="primary"
                        disabled={pendingTotal > 0}
                        onClick={() => void apply((e) => e, `Review accepted by ${reviewer.trim()}.`, 'ACCEPTED')}
                      >
                        Accept review
                      </button>
                      <button
                        disabled={pendingTotal > 0}
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
