import { useRef } from 'react'
import { COLORS, STATES, totals } from './model.ts'
import type { Behavior, DemoSession, Segment } from './model.ts'
import { Card } from './ui.tsx'

export function StateSummary({
  segments,
  duration,
}: {
  segments: Segment[]
  duration: number
}) {
  const sums = totals(segments)
  return (
    <Card
      title="Time per state"
      eyebrow="Behavior distribution"
      accessory={
        <span className="small muted">{duration.toFixed(1)} s total</span>
      }
    >
      <div className="state-bars">
        {STATES.filter((s) => sums[s] > 0).map((s) => (
          <div key={s} className="state-bar">
            <div className="bar-label">
              <span>
                <i style={{ background: COLORS[s] }} />
                {s}
              </span>
              <b>{sums[s].toFixed(1)} s</b>
            </div>
            <div className="bar-track">
              <div
                style={{
                  width: `${(sums[s] / duration) * 100}%`,
                  background: COLORS[s],
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

export function Ethogram({
  original,
  segments,
  duration,
  time,
  range,
  setRange,
  seek,
  stale,
  draft,
}: {
  original: Segment[]
  segments: Segment[]
  duration: number
  time: number
  range: [number, number]
  setRange: (r: [number, number]) => void
  seek: (t: number) => void
  stale: boolean
  draft?: Behavior
}) {
  const start = useRef<number | null>(null)
  const getTime = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return Math.max(
      0,
      Math.min(duration, ((e.clientX - rect.left) / rect.width) * duration),
    )
  }
  return (
    <Card
      title="Behavior timeline"
      eyebrow="Ethogram"
      accessory={
        stale ? (
          <span className="badge warning">Automatic labels stale</span>
        ) : (
          <span className="small muted">Drag to select a range</span>
        )
      }
    >
      <div className="ethogram-label">
        <span>Automatic</span>
        <span>Reviewed</span>
      </div>
      <svg
        className="ethogram"
        viewBox="0 0 1000 86"
        preserveAspectRatio="none"
        role="img"
        aria-label="Automatic and reviewed behavior timelines. Use playback and range fields for keyboard control."
        onPointerDown={(e) => {
          start.current = getTime(e)
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          if (start.current !== null)
            setRange([
              Math.min(start.current, getTime(e)),
              Math.max(start.current, getTime(e)),
            ])
        }}
        onPointerUp={(e) => {
          if (start.current === null) return
          const t = getTime(e)
          if (Math.abs(t - start.current) < 0.1) seek(t)
          else
            setRange([Math.min(start.current, t), Math.max(start.current, t)])
          start.current = null
        }}
        onPointerCancel={() => {
          start.current = null
        }}
      >
        {[original, segments].map((row, i) =>
          row.map((s, j) => (
            <rect
              key={`${i}-${j}`}
              x={(s.start_s / duration) * 1000}
              y={i * 44}
              width={((s.end_s - s.start_s) / duration) * 1000}
              height="34"
              fill={COLORS[s.state]}
            >
              <title>
                {s.state} · {s.start_s.toFixed(2)}–{s.end_s.toFixed(2)} s ·{' '}
                {s.source}
              </title>
            </rect>
          )),
        )}
        {draft && range.every(Number.isFinite) && range[0] < range[1] && (
          <rect
            x={(range[0] / duration) * 1000}
            y="44"
            width={((range[1] - range[0]) / duration) * 1000}
            height="34"
            fill={COLORS[draft]}
            stroke="#c6b4ff"
            strokeWidth="3"
            strokeDasharray="4 3"
          >
            <title>Draft: {draft} (not saved)</title>
          </rect>
        )}
        {range.every(Number.isFinite) && range[0] < range[1] && (
          <rect
            x={(range[0] / duration) * 1000}
            y="0"
            width={((range[1] - range[0]) / duration) * 1000}
            height="78"
            fill="#1d4ed822"
            stroke="#172d4c"
            strokeWidth="2"
            strokeDasharray="5 4"
          />
        )}
        <line
          x1={(time / duration) * 1000}
          x2={(time / duration) * 1000}
          y1="0"
          y2="82"
          stroke="#14253d"
          strokeWidth="3"
        />
      </svg>
      <div className="time-axis">
        {[0, 0.25, 0.5, 0.75, 1].map((n) => (
          <span key={n}>{(n * duration).toFixed(0)} s</span>
        ))}
      </div>
      <div className="legend">
        {STATES.map((s) => (
          <span key={s}>
            <i style={{ background: COLORS[s] }} />
            {s}
          </span>
        ))}
      </div>
      <p className="small muted">
        Selection: {range[0].toFixed(2)}–{range[1].toFixed(2)} s.{' '}
        {draft
          ? `The reviewed row previews ${draft} here until you apply it.`
          : 'The reviewed row includes manual labels.'}
      </p>
    </Card>
  )
}

export function Traces({
  session,
  time,
  seek,
  stale,
}: {
  session: DemoSession
  time: number
  seek: (t: number) => void
  stale: boolean
}) {
  const specs = [
    ['speed', 'Speed', 'px/s', 120, '#277bb2'],
    ['turning', 'Turning', '°/s', 40, '#9673b7'],
    ['depth', 'Depth', 'px', 220, '#74a79b'],
  ] as const
  return (
    <Card
      title="Movement traces"
      eyebrow="Illustrative measurements · read-only"
      accessory={
        stale ? (
          <span className="badge warning">Measurements stale</span>
        ) : (
          <span className="badge neutral">Sample data</span>
        )
      }
    >
      <div className="traces">
        {specs.map(([key, label, unit, max, color]) => (
          <div className="trace" key={key}>
            <div className="trace-title">
              <b>{label}</b>
              <span>{unit}</span>
            </div>
            <svg
              viewBox="0 0 900 100"
              preserveAspectRatio="none"
              role="img"
              aria-label={`${label} over time. Clicking seeks video; playback seek provides keyboard access.`}
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect()
                seek(((e.clientX - r.left) / r.width) * session.duration)
              }}
            >
              {[20, 50, 80].map((y) => (
                <line
                  key={y}
                  x1="0"
                  x2="900"
                  y1={y}
                  y2={y}
                  stroke="#e7edf3"
                  strokeDasharray="3 4"
                />
              ))}
              <polyline
                points={session.measurements
                  .map(
                    (m) =>
                      `${(m.t / session.duration) * 900},${92 - (m[key] / max) * 80}`,
                  )
                  .join(' ')}
                fill="none"
                stroke={color}
                strokeWidth="2.5"
              />
              <line
                x1={(time / session.duration) * 900}
                x2={(time / session.duration) * 900}
                y1="0"
                y2="100"
                stroke="#132c4c"
                strokeDasharray="4 4"
              />
            </svg>
          </div>
        ))}
      </div>
      <div className="time-axis">
        <span>0 s</span>
        <span>{session.duration} s</span>
      </div>
      <details className="measurement-table">
        <summary>View numerical sample measurements</summary>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Time (s)</th>
                <th>Speed (px/s)</th>
                <th>Turning (°/s)</th>
                <th>Depth (px)</th>
              </tr>
            </thead>
            <tbody>
              {session.measurements
                .filter((_, i) => i % 4 === 0)
                .map((m) => (
                  <tr key={m.t}>
                    <td>
                      <button className="text-button" onClick={() => seek(m.t)}>
                        {m.t.toFixed(1)}
                      </button>
                    </td>
                    <td>{m.speed.toFixed(1)}</td>
                    <td>{m.turning.toFixed(1)}</td>
                    <td>{m.depth.toFixed(1)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </details>
    </Card>
  )
}
