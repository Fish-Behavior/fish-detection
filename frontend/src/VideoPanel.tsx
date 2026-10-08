import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { frameAt, trailRuns, videoPoint } from './model.ts'
import type {
  Box,
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
  keypoint: string
  sceneDraft: Scene
  frameDraft: FrameValue
  sketch: (tool: EditTool, p: Point | Box) => void
  nudge: (tool: EditTool, dx: number, dy: number) => void
}
const PREVIEW = '#c6b4ff'
export default function VideoPanel(p: VideoProps) {
  const { session, edits, src, title, time, duration, videoRef, seek, tool } = p
  const [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState('1'),
    [error, setError] = useState('')
  const [layers, setLayers] = useState({
    point: true,
    trail: true,
    box: true,
    keypoints: true,
    waterline: true,
    roi: false,
  })
  const drag = useRef<Point | null>(null)
  const index = Math.max(
    0,
    Math.min(
      Math.round(time * session.overlay.fps),
      session.overlay.t.length - 1,
    ),
  )
  const hasTracking = session.overlay.t.length > 0
  const frame = frameAt(session, index, edits),
    scene = edits.scene ?? session.scene
  const finite = (...n: (number | null)[]) => n.every(Number.isFinite),
    validBox = (b: Box) => finite(...b) && b[2] > b[0] && b[3] > b[1],
    sameBox = (a: Box, b: Box) => a.every((v, i) => v === b[i])
  const d = p.frameDraft
  // Unsaved drafts preview whenever they differ from the saved value, whatever tool is selected.
  const draft = {
    waterline:
      finite(p.sceneDraft.waterline) &&
      p.sceneDraft.waterline !== scene.waterline,
    roi: validBox(p.sceneDraft.roi) && !sameBox(p.sceneDraft.roi, scene.roi),
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
  const locate = (e: React.PointerEvent<SVGSVGElement>) =>
    videoPoint(
      [e.clientX, e.clientY],
      e.currentTarget.getBoundingClientRect(),
      session.overlay.width,
      session.overlay.height,
    )
  const rectangle = (b: Box) => ({
    x: b[0],
    y: b[1],
    width: b[2] - b[0],
    height: b[3] - b[1],
  })
  return (
    <Card
      title="Tracking workspace"
      eyebrow="Video + prepds overlays"
      accessory={<span className="badge success">Session video</span>}
      className="video-card"
    >
      <div className="video-stage">
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
            aria-label="Fish overlays. With an edit tool selected, focus here and use arrow keys (Shift for 10 px) to move the draft."
            tabIndex={tool !== 'inspect' ? 0 : undefined}
            onKeyDown={(e) => {
              const step = e.shiftKey ? 10 : 1,
                move = {
                  ArrowLeft: [-step, 0],
                  ArrowRight: [step, 0],
                  ArrowUp: [0, -step],
                  ArrowDown: [0, step],
                }[e.key]
              if (tool === 'inspect' || !move) return
              e.preventDefault()
              p.nudge(tool, move[0], move[1])
            }}
            onPointerDown={(e) => {
              if (tool === 'inspect') return
              const q = locate(e)
              if (!q) return
              drag.current = q
              e.currentTarget.setPointerCapture(e.pointerId)
              if (!['roi', 'box'].includes(tool)) p.sketch(tool, q)
            }}
            onPointerMove={(e) => {
              if (drag.current && !['roi', 'box'].includes(tool)) {
                const q = locate(e)
                if (q) p.sketch(tool, q)
              }
            }}
            onPointerUp={(e) => {
              if (drag.current && ['roi', 'box'].includes(tool)) {
                const q = locate(e)
                if (q)
                  p.sketch(tool, [
                    Math.min(drag.current[0], q[0]),
                    Math.min(drag.current[1], q[1]),
                    Math.max(drag.current[0], q[0]),
                    Math.max(drag.current[1], q[1]),
                  ])
              }
              drag.current = null
            }}
            onPointerCancel={() => {
              drag.current = null
            }}
          >
            {layers.roi && (
              <rect
                {...rectangle(scene.roi)}
                fill="none"
                stroke="#8ccb9d"
                strokeWidth="1.5"
                strokeDasharray="5 4"
              />
            )}
            {layers.waterline && (
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
            {hasTracking && !frame.detected && (
              <text x="45" y="94" fill="white" fontSize="12">
                Fish not detected
              </text>
            )}
            {draft.waterline && (
              <line
                x1="0"
                x2={session.overlay.width}
                y1={p.sceneDraft.waterline}
                y2={p.sceneDraft.waterline}
                stroke={PREVIEW}
                strokeWidth="2"
              />
            )}
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
          </svg>
        }
        <span className="video-stamp">TRACKING OVERLAYS</span>
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
      <div className="frame-control">
        <button
          onClick={() => {
            videoRef.current?.pause()
            seek(time - 1 / session.overlay.fps)
          }}
        >
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
        <button
          onClick={() => {
            videoRef.current?.pause()
            seek(time + 1 / session.overlay.fps)
          }}
        >
          Next frame
        </button>
        <span className="small muted">
          Frame {index} · {session.overlay.width} × {session.overlay.height} ·{' '}
          {session.overlay.fps} fps
        </span>
      </div>
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      {
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
                    point: 'Track point',
                    trail: '2 s trail',
                    box: 'Box',
                    keypoints: 'Keypoints',
                    waterline: 'Waterline',
                    roi: 'ROI',
                  } as Record<string, string>
                )[key]
              }
            </label>
          ))}
        </div>
      }
      <p className="video-note">
        {tool === 'inspect'
          ? 'Stored x/y point + optional detector overlays, using prepds field conventions.'
          : `Edit ${tool}: ${['box', 'roi'].includes(tool) ? 'drag a rectangle' : 'click or drag on the video'}, or focus the video and use arrow keys (Shift = 10 px). Purple shows the unsaved draft; Apply in the review panel to save.`}
      </p>
    </Card>
  )
}
