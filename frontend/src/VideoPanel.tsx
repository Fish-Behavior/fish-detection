import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { adjacentFrameTime, boxHandle, dragBox, frameAt, frameIndexAt, trailRuns, videoPoint } from './model.ts'
import type {
  Box,
  BoxHandle,
  SessionData,
  Edits,
  FrameValue,
  Point,
  Scene,
} from './model.ts'
import { Card, Icon } from './ui.tsx'

export type EditTool =
  'inspect' | 'waterline' | 'roi' | 'point' | 'box' | 'keypoint'
export interface VideoProps {
  reviewMode: boolean
  markRange: (boundary: 'start' | 'end') => void
  session: SessionData
  edits: Edits
  src: string
  title: string
  time: number
  duration: number
  setTime: (t: number) => void
  setDuration: (d: number) => void
  videoRef: RefObject<HTMLVideoElement | null>
  seek: (t: number) => void
  tool: EditTool
  setTool?: (t: EditTool) => void
  keypoint: string
  sceneDraft: Scene
  frameDraft: FrameValue
  sketch: (tool: EditTool, p: Point | Box) => void
  nudge: (tool: EditTool, dx: number, dy: number) => void
}
const PREVIEW = '#c6b4ff'
const RECT_HINT = 'drag a new one, drag inside to move it, or drag a corner to resize.'
const TOOLS: { id: EditTool; label: string; key: string; hint: string; detector?: boolean }[] = [
  { id: 'inspect', label: 'Inspect', key: 'v', hint: 'Play and scrub. Pick a tool to correct what you see.' },
  { id: 'point', label: 'Re-point fish', key: 'p', hint: 'Click the fish to move the track point; drag to fine-tune.' },
  { id: 'box', label: 'Box', key: 'b', hint: `Detector box: ${RECT_HINT}`, detector: true },
  { id: 'keypoint', label: 'Keypoint', key: 'k', hint: 'Click where the selected keypoint belongs (choose it in Frame correction).', detector: true },
  { id: 'roi', label: 'ROI', key: 'r', hint: `Region of interest: ${RECT_HINT}` },
  { id: 'waterline', label: 'Waterline', key: 'w', hint: 'Click or drag at the water surface.' },
]
const CURSORS: Record<BoxHandle, string> = { new: 'crosshair', move: 'move', nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize' }
export default function VideoPanel(p: VideoProps) {
  const { session, edits, src, title, time, duration, videoRef, seek, reviewMode } = p
  const tool = reviewMode ? p.tool : 'inspect'
  const [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState('1'),
    [error, setError] = useState('')
  const defaultLayers = {
    point: true,
    trail: true,
    box: true,
    keypoints: true,
    waterline: true,
    roi: false,
  }
  const [reviewLayers, setLayers] = useState(defaultLayers)
  const layers = reviewMode ? reviewLayers : defaultLayers
  const stage = useRef<HTMLDivElement>(null)
  const drag = useRef<{ start: Point; handle: BoxHandle; origin: Box | null; slack: number } | null>(null)
  const index = frameIndexAt(session, time)
  const hasTracking = session.overlay.t.length > 0
  const frame = frameAt(session, index, edits),
    scene = edits.scene ?? session.scene
  const finite = (...n: (number | null)[]) => n.every(Number.isFinite),
    validBox = (b: Box) => finite(...b) && b[2] > b[0] && b[3] > b[1],
    sameBox = (a: Box, b: Box) => a.every((v, i) => v === b[i])
  const d = reviewMode ? p.frameDraft : frame
  const sceneDraft = reviewMode ? p.sceneDraft : scene
  // Unsaved drafts preview whenever they differ from the saved value, whatever tool is selected.
  const draft = {
    waterlineCleared: sceneDraft.waterline === null && scene.waterline !== null,
    waterline:
      finite(sceneDraft.waterline) &&
      sceneDraft.waterline !== scene.waterline,
    roi: validBox(sceneDraft.roi) && !sameBox(sceneDraft.roi, scene.roi),
    box:
      !!d.box && validBox(d.box) && !(frame.box && sameBox(d.box, frame.box)),
    point:
      d.detected && finite(d.x, d.y) && (d.x !== frame.x || d.y !== frame.y),
    keypoints: Object.entries(d.keypoints)
      .filter(
        ([k, [x, y]]) =>
          finite(x, y) &&
          (!frame.keypoints[k] ||
            x !== frame.keypoints[k][0] ||
            y !== frame.keypoints[k][1]),
      )
      .map(([k, [x, y]]) => [k, x, y] as [string, number, number]),
    missing: !d.detected && frame.detected,
  }
  useEffect(() => {
    setPlaying(false)
    setError('')
    setSpeed('1')
  }, [src])
  useEffect(() => {
    const video = videoRef.current
    if (!video || !playing) return
    let id = 0
    const loop = () => {
      p.setTime(video.currentTime)
      id = requestAnimationFrame(loop)
    }
    id = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(id)
  }, [playing, p.setTime, videoRef])
  useEffect(() => {
    if (tool !== 'inspect') videoRef.current?.pause()
  }, [tool, videoRef])
  // A drag never carries over to another tool or frame.
  useEffect(() => {
    drag.current = null
  }, [tool, index])
  const locate = (e: React.PointerEvent<SVGSVGElement>, clamp = false) =>
    videoPoint(
      [e.clientX, e.clientY],
      e.currentTarget.getBoundingClientRect(),
      session.overlay.width,
      session.overlay.height,
      clamp,
    )
  const rectTool = tool === 'roi' || tool === 'box'
  // The rectangle the ROI and Box tools reshape: the draft, which starts as the saved value.
  const editable = tool === 'roi' ? p.sceneDraft.roi : tool === 'box' ? p.frameDraft.box : null
  const handleSize = Math.max(session.overlay.width, session.overlay.height) / 90
  // Corner grab distance: 12 screen pixels, in video pixels.
  const reach = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    return 12 / (Math.min(r.width / session.overlay.width, r.height / session.overlay.height) || 1)
  }
  const stepFrame = (direction: -1 | 1) => {
    videoRef.current?.pause()
    seek(adjacentFrameTime(session, time, direction))
  }
  const rectangle = (b: Box) => ({
    x: b[0],
    y: b[1],
    width: b[2] - b[0],
    height: b[3] - b[1],
  })
  return (
    <Card
      title={reviewMode ? 'Correct the tracking' : 'Recording'}
      className="video-card"
    >
      {reviewMode && <div className="tool-bar">
        <div role="toolbar" aria-label="Correction tools" className="tool-buttons">
          {TOOLS.filter((t) => !t.detector || session.overlay.has_detector).map((t) => (
            <button type="button" key={t.id} aria-pressed={tool === t.id} onClick={() => {
              p.setTool?.(t.id)
              // Hand the keyboard to the player, where shortcuts and nudges work.
              stage.current?.focus({ preventScroll: true })
            }}>
              {t.label}
              <kbd aria-hidden="true">{t.key.toUpperCase()}</kbd>
            </button>
          ))}
        </div>
        <p className="tool-hint" aria-live="polite">
          {TOOLS.find((t) => t.id === tool)?.hint}
          {tool !== 'inspect' && ' Purple is unsaved and autosaves in this browser.'}
        </p>
      </div>}
      <div className="video-stage" ref={stage} tabIndex={reviewMode ? 0 : undefined}
        role="group" aria-label={reviewMode ? 'Review player. Letter keys pick a tool, comma and period step frames, arrow keys seek or nudge, brackets mark the range.' : 'Session player'}
        onKeyDown={(e) => {
          if (!reviewMode || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey ||
            (e.target as Element).closest('input, select, textarea, button, [contenteditable="true"]')) return
          const shortcut = TOOLS.find((t) => t.key === e.key.toLowerCase() && (!t.detector || session.overlay.has_detector))
          if (e.key === '[' || e.key === ']') {
            e.preventDefault(); p.markRange(e.key === '[' ? 'start' : 'end')
          } else if (e.key === ',' || e.key === '.') {
            e.preventDefault(); stepFrame(e.key === ',' ? -1 : 1)
          } else if (p.setTool && (shortcut || e.key === 'Escape')) {
            e.preventDefault(); p.setTool(shortcut?.id ?? 'inspect')
          } else if (tool === 'inspect' && ['ArrowLeft', 'ArrowRight'].includes(e.key)) {
            e.preventDefault(); seek(time + (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1))
          } else if (tool !== 'inspect' && e.key.startsWith('Arrow')) {
            const step = e.shiftKey ? 10 : 1
            const [dx, dy] = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key] ?? [0, 0]
            e.preventDefault(); p.nudge(tool, dx, dy)
          }
        }}>
        <video
          ref={videoRef}
          src={src}
          preload="auto"
          playsInline
          aria-label={title}
          onLoadedMetadata={(e) => {
            p.setDuration(
              Number.isFinite(e.currentTarget.duration)
                ? e.currentTarget.duration
                : 0,
            )
            e.currentTarget.playbackRate = Number(speed)
          }}
          onTimeUpdate={(e) => p.setTime(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={() =>
            setError(
              'This browser cannot play this file. Choose a browser-supported MP4 or WebM.',
            )
          }
        />
        {
          <svg
            viewBox={`0 0 ${session.overlay.width} ${session.overlay.height}`}
            className={`video-overlay ${tool !== 'inspect' ? 'editing' : ''}`}
            role="img"
            aria-label="Fish overlays. With an edit tool selected, arrow keys on the player move the draft (Shift for 10 px)."
            onPointerDown={(e) => {
              if (tool === 'inspect' || e.button !== 0) return
              // Rectangle corners often sit on the picture edge, so a press just outside it still grabs them.
              const q = locate(e, rectTool)
              if (!q) return
              const handle = rectTool ? boxHandle(q, editable, reach(e)) : 'new'
              // Movement under 4 screen pixels is click jitter, not a drag.
              drag.current = { start: q, handle, origin: editable && [...editable], slack: reach(e) / 3 }
              e.currentTarget.setPointerCapture(e.pointerId)
              // A rectangle waits for movement, so a stray click never collapses it.
              if (!rectTool) p.sketch(tool, q)
            }}
            onPointerMove={(e) => {
              const d = drag.current
              if (!d) {
                const q = rectTool && locate(e, true)
                e.currentTarget.style.cursor = rectTool ? CURSORS[q ? boxHandle(q, editable, reach(e)) : 'new'] : ''
                return
              }
              const q = locate(e, true)
              if (!q) return
              if (!rectTool) p.sketch(tool, q)
              else if (Math.hypot(q[0] - d.start[0], q[1] - d.start[1]) >= d.slack) {
                const box = dragBox(d.handle, d.origin, d.start, q, session.overlay.width, session.overlay.height)
                // A flat rectangle (dragged along an edge) is never a usable ROI or box.
                if (box[2] - box[0] >= 1 && box[3] - box[1] >= 1) p.sketch(tool, box)
              }
            }}
            onPointerUp={() => {
              drag.current = null
            }}
            onPointerCancel={() => {
              drag.current = null
            }}
          >
            {(layers.roi || tool === 'roi') && (
              <rect
                {...rectangle(scene.roi)}
                fill="none"
                stroke="#8ccb9d"
                strokeWidth="1.5"
                strokeDasharray="5 4"
              />
            )}
            {layers.waterline && scene.waterline !== null && !draft.waterlineCleared && (
              <line
                x1="0"
                x2={session.overlay.width}
                y1={scene.waterline}
                y2={scene.waterline}
                stroke="#67b7ff"
                strokeWidth="1.5"
                strokeDasharray="7 4"
              />
            )}
            {layers.trail &&
              trailRuns(session, edits, index).map((run, i) => (
                <polyline
                  key={i}
                  points={run.map((p) => p.join(',')).join(' ')}
                  fill="none"
                  stroke="#ffa85b"
                  strokeWidth="1.6"
                />
              ))}
            {frame.detected && (
              <>
                {layers.box && frame.box && (
                  <rect
                    {...rectangle(frame.box)}
                    fill="none"
                    stroke="#ffd36a"
                    strokeWidth="1.2"
                  />
                )}
                {layers.keypoints && (
                  <>
                    {['snout', 'tail_base', 'tail_tip']
                      .slice(0, 2)
                      .map((a, i) => {
                        const b = ['tail_base', 'tail_tip'][i]
                        if (!frame.keypoints[a] || !frame.keypoints[b])
                          return null
                        return (
                          <line
                            key={a}
                            x1={frame.keypoints[a][0]}
                            y1={frame.keypoints[a][1]}
                            x2={frame.keypoints[b][0]}
                            y2={frame.keypoints[b][1]}
                            stroke="#e9a7ee"
                            strokeWidth="1"
                          />
                        )
                      })}
                    {Object.entries(frame.keypoints).map(([name, [x, y]]) => (
                      <circle
                        key={name}
                        cx={x}
                        cy={y}
                        r="2.8"
                        fill="#e9a7ee"
                        stroke="#fff"
                        strokeWidth=".5"
                      >
                        <title>{name}</title>
                      </circle>
                    ))}
                  </>
                )}
                {layers.point && frame.x !== null && frame.y !== null && (
                  <circle
                    cx={frame.x}
                    cy={frame.y}
                    r="4"
                    fill="#ff606d"
                    stroke="white"
                    strokeWidth="1.1"
                  />
                )}
              </>
            )}
            {hasTracking && !frame.detected && !draft.point && (
              <text x="45" y="94" fill="white" fontSize="12">
                Fish not detected
              </text>
            )}
            {draft.waterline && (
              <line
                x1="0"
                x2={session.overlay.width}
                y1={p.sceneDraft.waterline ?? undefined}
                y2={p.sceneDraft.waterline ?? undefined}
                stroke={PREVIEW}
                strokeWidth="2"
              />
            )}
            {draft.waterlineCleared && <text x="45" y="130" fill={PREVIEW} fontSize="12">Draft: waterline cleared</text>}
            {draft.roi && (
              <rect
                {...rectangle(p.sceneDraft.roi)}
                fill="#c6b4ff11"
                stroke={PREVIEW}
                strokeWidth="2"
                strokeDasharray="4 3"
              />
            )}
            {draft.box && p.frameDraft.box && (
              <rect
                {...rectangle(p.frameDraft.box)}
                fill="none"
                stroke={PREVIEW}
                strokeWidth="2"
                strokeDasharray="4 3"
              />
            )}
            {draft.point && (
              <circle
                cx={p.frameDraft.x!}
                cy={p.frameDraft.y!}
                r="6"
                fill="none"
                stroke={PREVIEW}
                strokeWidth="2"
              />
            )}
            {draft.keypoints.map(([name, x, y]) => (
              <circle
                key={name}
                cx={x}
                cy={y}
                r="6"
                fill="none"
                stroke={PREVIEW}
                strokeWidth="2"
              />
            ))}
            {draft.missing && (
              <text x="45" y="112" fill={PREVIEW} fontSize="12">
                Draft: fish marked missing
              </text>
            )}
            {rectTool && editable && editable.every(Number.isFinite) &&
              [[0, 1], [2, 1], [0, 3], [2, 3]].map(([xi, yi]) => (
                <rect
                  key={`${xi}${yi}`}
                  className="box-handle"
                  x={editable[xi] - handleSize / 2}
                  y={editable[yi] - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                />
              ))}
          </svg>
        }
      </div>
      <div className="playback">
        <button
          className="play-button"
          aria-label={playing ? 'Pause video' : 'Play video'}
          disabled={!!error}
          onClick={() => {
            const v = videoRef.current
            if (!v) return
            if (v.paused)
              void v
                .play()
                .catch(() =>
                  setError('Playback could not start. Try another file.'),
                )
            else v.pause()
          }}
        >
          <Icon name={playing ? 'pause' : 'play'} />
        </button>
        <span className="clock">
          {time.toFixed(2)} <span>/ {duration.toFixed(2)} s</span>
        </span>
        <input
          type="range"
          aria-label="Seek video"
          min="0"
          max={duration || 1}
          step={1 / session.overlay.fps}
          value={Math.min(time, duration)}
          onChange={(e) => seek(Number(e.target.value))}
        />
        <select
          aria-label="Playback speed"
          value={speed}
          onChange={(e) => {
            setSpeed(e.target.value)
            if (videoRef.current)
              videoRef.current.playbackRate = Number(e.target.value)
          }}
        >
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="2">2×</option>
        </select>
      </div>
      {reviewMode && <div className="frame-control">
        <button onClick={() => stepFrame(-1)}>
          Previous frame
        </button>
        <label>
          Time (s)
          <input
            type="number"
            aria-label="Playback time in seconds"
            value={Number(time.toFixed(3))}
            step={1 / session.overlay.fps}
            min="0"
            max={duration}
            onChange={(e) => seek(Number(e.target.value))}
          />
        </label>
        <button onClick={() => stepFrame(1)}>
          Next frame
        </button>
        <span className="small muted">
          Frame {index} · {session.overlay.width} × {session.overlay.height} ·{' '}
          {session.overlay.fps} fps
        </span>
      </div>}
      {reviewMode && <div className="button-row range-markers">
        <button onClick={() => p.markRange('start')}>Mark range start [</button>
        <button onClick={() => p.markRange('end')}>Mark range end ]</button>
      </div>}
      {reviewMode && <details className="shortcut-help">
        <summary>Keyboard shortcuts</summary>
        <dl>
          <dt>{TOOLS.filter((t) => !t.detector || session.overlay.has_detector).map((t) => t.key.toUpperCase()).join(' ')}</dt>
          <dd>Pick a tool; Esc returns to Inspect</dd>
          <dt>, and .</dt><dd>Previous or next frame</dd>
          <dt>Arrow keys</dt><dd>Inspect: seek 1 s (Shift 10 s). With a tool: nudge the draft 1 px (Shift 10 px)</dd>
          <dt>[ and ]</dt><dd>Mark range start or end</dd>
        </dl>
        <p className="small muted">Click the video first so it has keyboard focus. Browser frame seeking is approximate.</p>
      </details>}
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      {reviewMode &&
        <div className="overlay-toggles">
          {Object.entries(layers).map(([key, value]) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={value}
                onChange={(e) =>
                  setLayers({ ...layers, [key]: e.target.checked })
                }
              />
              <span className={`layer-dot ${key}`} />
              {
                (
                  {
                    point: 'Fish position',
                    trail: 'Path, last 2 s',
                    box: 'Detection box',
                    keypoints: 'Body points',
                    waterline: 'Water surface',
                    roi: 'Region of interest',
                  } as Record<string, string>
                )[key]
              }
            </label>
          ))}
        </div>
      }
      {!reviewMode && hasTracking && <div className="legend overlay-legend" aria-label="What the marks on the video show">
        <span><i className="layer-dot point" />Fish position</span>
        <span><i className="layer-dot trail" />Path, last 2 seconds</span>
        {session.overlay.has_detector && <span><i className="layer-dot box" />Detection box</span>}
        {session.overlay.has_detector && <span><i className="layer-dot keypoints" />Body points</span>}
        <span><i className="layer-dot waterline" />Water surface</span>
      </div>}
    </Card>
  )
}
