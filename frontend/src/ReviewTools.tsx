import { useEffect, useState } from 'react'
import { explainSecond, loadReviewTools } from './api.ts'
import type { ReviewToolsData, SecondExplanation, SessionData } from './model.ts'
import { Card } from './ui.tsx'

export default function ReviewTools({ session, time, corrected, selectRange }: {
  session: SessionData
  time: number
  corrected: boolean
  selectRange: (start: number, end: number) => void
}) {
  const [tools, setTools] = useState<ReviewToolsData | null>(null)
  const [explanation, setExplanation] = useState<SecondExplanation | null>(null)
  const [toolsError, setToolsError] = useState('')
  const [whyError, setWhyError] = useState('')
  const second = Math.max(0, Math.min(Math.floor(time), Math.ceil(session.duration) - 1))
  useEffect(() => {
    let alive = true
    setTools(null); setToolsError('')
    loadReviewTools(session.video_id, session.baseline).then(
      (value) => { if (alive) setTools(value) },
      (error) => { if (alive) setToolsError(error.message) },
    )
    return () => { alive = false }
  }, [session.video_id, session.baseline])
  useEffect(() => {
    let alive = true
    setExplanation(null); setWhyError('')
    const timer = setTimeout(() => {
      explainSecond(session.video_id, second, session.baseline).then(
        (value) => { if (alive) setExplanation(value) },
        (error) => { if (alive) setWhyError(error.message) },
      )
    }, 150)
    return () => { alive = false; clearTimeout(timer) }
  }, [session.video_id, session.baseline, second])
  return <div className="prepds-review-tools">
    <Card title="Prepds review flags" eyebrow="Original pipeline output">
      {toolsError && <p className="inline-error" role="alert">{toolsError}</p>}
      {!tools && !toolsError && <p className="muted" role="status">Loading review data…</p>}
      {tools && <>
        <dl className="review-metadata">
          <dt>Subject / sex</dt><dd>{tools.manifest.subject_id} / {tools.manifest.sex}</dd>
          <dt>Exposure</dt><dd>{tools.manifest.compound} · {tools.manifest.concentration_mM} mM</dd>
          <dt>Calibration</dt><dd>{tools.manifest.calibration_profile_version}</dd>
          <dt>Pipeline</dt><dd>{tools.manifest.pipeline_version} · {tools.manifest.processed_at}</dd>
          <dt>Prepds status</dt><dd>{tools.manifest.review_status} · {tools.manifest.edit_count} legacy edits{tools.manifest.reviewer && ` · ${tools.manifest.reviewer}`}</dd>
        </dl>
        <ul className="review-hints">
          {tools.manifest.review_flags.map((flag, i) => <li key={`flag-${i}`}>
            <button className="text-button" onClick={() => selectRange(flag.start_s, flag.end_s)}>
              Check {flag.start_s.toFixed(2)}–{flag.end_s.toFixed(2)} s
            </button>
            <span>{flag.kind}: {flag.message}</span>
          </li>)}
        </ul>
        {tools.manifest.review_flags.length === 0 && <p className="small muted">No pipeline review flags.</p>}
        <h3>Possible Listing/LORR</h3>
        <p className="small muted">Advisory hints from the original crop classifier, not confirmed labels.{corrected && ' These hints do not reflect your corrections.'}</p>
        {tools.listing_flags_error && <p className="inline-error" role="alert">Listing hints unavailable: {tools.listing_flags_error}</p>}
        <ul className="review-hints">
          {tools.listing_flags.map((flag, i) => <li key={`listing-${i}`}>
            <button className="text-button" onClick={() => selectRange(flag.start_s, flag.end_s)}>
              Inspect {flag.start_s.toFixed(2)}–{flag.end_s.toFixed(2)} s
            </button>
            <span>Score {flag.max_score.toFixed(3)} · {flag.n_samples} samples</span>
          </li>)}
        </ul>
        {!tools.listing_flags_error && tools.listing_flags.length === 0 && <p className="small muted">No Listing hints available.</p>}
      </>}
    </Card>
    <Card title="Why this label?" eyebrow="Original pipeline label explanation" accessory={<span className="small muted">Second {second}</span>}>
      <p className="small muted">This replays the stored track with its calibration profile.{corrected && ' It does not explain saved corrections or staged labels.'}</p>
      {whyError && <p className="inline-error" role="alert">{whyError}</p>}
      {!explanation && !whyError && <p className="muted" role="status">Loading explanation…</p>}
      {explanation && <>
        <p><b>{explanation.stored_state}</b> · {explanation.stored_source} · {explanation.start_s.toFixed(2)}–{explanation.end_s.toFixed(2)} s</p>
        <p className="small">Fish detected in {explanation.n_detected} of {explanation.n_frames} frames.
          {explanation.speed_median_px_per_s !== null && ` Median speed: ${explanation.speed_median_px_per_s.toFixed(1)} px/s.`}
          {explanation.y_min_px !== null && ` Minimum y: ${explanation.y_min_px.toFixed(1)} px.`}
        </p>
        {Object.keys(explanation.votes).length > 0 && <p className="small">Frame votes: {Object.entries(explanation.votes).map(([state, count]) => `${count} ${state}`).join(', ')}.</p>}
        {explanation.matches_stored === false && <p className="notice">Replayed automatic label: {explanation.auto_state}. The stored label differs; a manual label, profile change or track rounding can cause this.</p>}
        {!explanation.thresholds_available && <p className="inline-error">Rules unavailable: {explanation.thresholds_error ?? 'Calibration thresholds are missing.'}</p>}
        <ul className="review-rules">
          {explanation.rules.map((rule) => <li key={rule.state}>
            <b>{rule.state}{rule.wins ? ' · wins' : ''} · {rule.frames} frames</b>
            <span>{rule.detail}</span>
          </li>)}
        </ul>
      </>}
    </Card>
  </div>
}
