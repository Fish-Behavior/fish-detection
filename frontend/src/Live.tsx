import { useEffect, useRef, useState } from 'react'
import { endLive, listCameras, livePreviewUrl, liveStatus, startLive } from './api.ts'
import { COLORS, secondRuns, sessionName, sessionStamp, validateReviewer } from './model.ts'
import type { Camera, LiveStatus } from './model.ts'
import { Card, Icon } from './ui.tsx'

const REVIEWER_KEY = 'fishlab:live-reviewer'
const elapsed = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`

// A camera is one more video source: while recording only the current behavior is shown; End saves the session
// through the same analysis as a video file, then the app opens it like any recording.
export default function LiveView({ onFinished, onClose }: { onFinished: (videoId: string) => void; onClose: () => void }) {
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [cameras, setCameras] = useState<Camera[] | null>(null) // null while looking
  const [camera, setCamera] = useState(0)
  const [reviewer, setReviewer] = useState(() => { try { return localStorage.getItem(REVIEWER_KEY) ?? '' } catch { return '' } })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [frame, setFrame] = useState(0)
  const [name, setName] = useState('')
  // Only a session started, resumed or ended in this view opens its result when it finishes.
  const following = useRef(false)

  const findCameras = async () => {
    setCameras(null); setError('')
    try {
      const found = (await listCameras()).cameras
      setCameras(found)
      if (found[0]) setCamera(found[0].index)
    } catch (e) { setCameras([]); setError((e as Error).message) }
  }
  useEffect(() => {
    liveStatus().then((s) => {
      setStatus(s)
      if (s.active) following.current = true // resume after a page reload
      else void findCameras()
    }, (e) => setError(e.message))
  }, [])
  const session = status && 'video_id' in status ? status : null
  const phase = session?.phase
  useEffect(() => {
    if (!session || phase === 'finished' || phase === 'failed') return
    const poll = setInterval(() => { liveStatus().then(setStatus, (e) => setError(e.message)) }, 500)
    return () => clearInterval(poll)
  }, [session?.video_id, phase])
  useEffect(() => {
    if (phase === 'finished' && following.current && session) { following.current = false; onFinished(session.video_id) }
  }, [phase])
  useEffect(() => { // a session resumed on mount skipped the camera search; run it once that session is over
    if ((phase === 'failed' || phase === 'finished') && cameras === null) void findCameras()
  }, [phase])

  const start = async () => {
    try { validateReviewer(reviewer) } catch (e) { setError((e as Error).message); return }
    try { localStorage.setItem(REVIEWER_KEY, reviewer.trim()) } catch { /* private window: keep it for this visit only */ }
    setBusy(true); setError('')
    try { setStatus(await startLive(camera, reviewer.trim(), sessionName(name) || undefined)); following.current = true } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  const end = async () => {
    setBusy(true); setError('')
    try { setStatus(await endLive()); following.current = true } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  const open = session && phase !== 'finished' && phase !== 'failed'
  const recording = phase === 'running' || phase === 'stopped'
  const runs = secondRuns(session?.seconds ?? [])
  const width = Math.max(session?.seconds.length ?? 0, 60)

  return (
    <main className="live-page">
      <div className="page-heading">
        <div>
          <h1>Live camera</h1>
          <p className="subtitle">Watch one fish live. While recording only the current behavior is shown; End saves the session and analyzes it like a video file, then shows the drug prediction.</p>
        </div>
        {!open && <button onClick={onClose}>Back to recordings</button>}
      </div>
      {error && <p className="inline-error" role="alert">{error}</p>}

      {!open && (
        <Card title="Start a live session" className="live-start">
          {phase === 'failed' && <p className="inline-error">{session?.message || 'The live session could not be saved.'}</p>}
          {cameras === null ? <p className="muted">Looking for cameras…</p> : cameras.length === 0 ? (
            <div className="live-empty">
              <p><b>No cameras detected.</b> Connect a USB camera to this computer, then look again.</p>
              <button onClick={() => void findCameras()}><Icon name="reset" />Look again</button>
            </div>
          ) : (
            <label className="field">
              Camera
              <select value={camera} onChange={(e) => setCamera(Number(e.target.value))}>
                {cameras.map((c) => <option key={c.index} value={c.index}>{c.name} ({c.width}×{c.height})</option>)}
              </select>
            </label>
          )}
          <label className="field">
            Session name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={status?.suggested_name ?? 'Video1'} maxLength={60} aria-describedby="session-name-hint" />
          </label>
          <p id="session-name-hint" className="small muted live-name-hint">Saved as <b>{sessionName(name) || (status?.suggested_name ?? 'Video1')}_{sessionStamp(new Date())}</b> (the date and time are added when you start). Letters, digits and dashes.</p>
          <label className="field">
            Reviewer
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Your name" maxLength={100} />
          </label>
          <button className="primary" disabled={busy || !cameras?.length} onClick={() => void start()}>Start recording</button>
        </Card>
      )}

      {open && session && (
        <div className="live-grid">
          <section className="live-video" aria-label="Live camera preview">
            {recording && session.frame_count > 0
              ? <img src={livePreviewUrl(frame)} alt="Live camera frame, the tracked fish marked with a green cross"
                  onLoad={() => setTimeout(() => setFrame((n) => n + 1), 150)} onError={() => setTimeout(() => setFrame((n) => n + 1), 500)} />
              : <p>{phase === 'finishing' ? 'Saving and analyzing…' : 'Opening the camera…'}</p>}
          </section>
          <Card title="Current behavior" accessory={<span className="badge neutral">{elapsed(session.t_sec)}</span>}>
            <div className="live-state" style={{ borderColor: session.state ? COLORS[session.state] : 'var(--line)' }} aria-live="polite">
              <span className="live-swatch" style={{ background: session.state ? COLORS[session.state] : 'var(--line)' }} />
              <strong>{session.state ?? 'Waiting for frames'}</strong>
            </div>
            <p className="small muted">{session.detected ? 'Fish detected.' : 'Fish not found in this frame.'} A state such as Freezing/Drift shows once it has lasted its minimum time; the saved labels are worked out again when you end the session.</p>
            <svg className="live-strip" viewBox={`0 0 ${width} 10`} preserveAspectRatio="none" role="img" aria-label="Behavior per second so far">
              {runs.map((r) => <rect key={r.start} x={r.start} width={r.end - r.start} y={0} height={10} fill={COLORS[r.state]}><title>{`${elapsed(r.start)}–${elapsed(r.end)} ${r.state}`}</title></rect>)}
            </svg>
            {phase === 'stopped' && <p className="notice">The camera stopped sending frames. End the session to save what was recorded.</p>}
            {phase === 'error' && <p className="inline-error">{session.message} End the session to close it.</p>}
            {phase === 'finishing'
              ? <p className="notice">Saving the session and running the analysis…</p>
              : <button className="primary" disabled={busy} onClick={() => void end()}>End and analyze</button>}
          </Card>
        </div>
      )}
    </main>
  )
}
