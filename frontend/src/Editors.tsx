import type { Dispatch, SetStateAction } from 'react'
import { emptyEdits, frameAt, STATES } from './model.ts'
import type {
  Behavior,
  Box,
  SessionData,
  Edits,
  FrameValue,
  LabelEdit,
  Scene,
} from './model.ts'
import type { EditTool } from './VideoPanel.tsx'
import { Card, Icon } from './ui.tsx'

function NumberField({
  label,
  value,
  onChange,
  max,
}: {
  label: string
  value: number
  onChange: (n: number) => void
  max?: number
}) {
  return (
    <label className="field">
      {label}
      <input
        type="number"
        step="any"
        min="0"
        max={max}
        value={Number.isFinite(value) ? Number(value.toFixed(3)) : ''}
        onChange={(e) =>
          onChange(e.target.value === '' ? NaN : Number(e.target.value))
        }
      />
    </label>
  )
}
function BoxFields({
  box,
  set,
  prefix,
  width,
  height,
}: {
  box: Box
  set: (b: Box) => void
  prefix: string
  width: number
  height: number
}) {
  return (
    <div className="field-grid four">
      {['Left', 'Top', 'Right', 'Bottom'].map((label, i) => (
        <NumberField
          key={label}
          label={`${prefix} ${label}`}
          value={box[i]}
          max={i % 2 ? height : width}
          onChange={(n) => {
            const next: Box = [...box]
            next[i] = n
            set(next)
          }}
        />
      ))}
    </div>
  )
}
export interface EditorProps {
  session: SessionData
  edits: Edits
  reviewer: string
  setReviewer: (s: string) => void
  tool: EditTool
  setTool: (t: EditTool) => void
  keypoint: string
  setKeypoint: (k: string) => void
  scene: Scene
  setScene: Dispatch<SetStateAction<Scene>>
  frame: FrameValue
  setFrame: Dispatch<SetStateAction<FrameValue>>
  frameIndex: number
  range: [number, number]
  setRange: (r: [number, number]) => void
  state: Behavior
  setState: (s: Behavior) => void
  saveScene: () => void
  saveFrame: () => void
  saveLabels: () => void
  resetScene: () => void
  resetFrame: () => void
  resetLabels: () => void
}
export default function Editors(
  p: Omit<
    EditorProps,
    'range' | 'setRange' | 'state' | 'setState' | 'saveLabels' | 'resetLabels'
  >,
) {
  const { session, frame, scene } = p,
    { width, height } = session.overlay
  const kp = frame.keypoints[p.keypoint]
  const markMissing = (missing: boolean) =>
    p.setFrame((f) => ({
      ...f,
      detected: !missing,
      x: missing ? null : (f.x ?? width / 2),
      y: missing ? null : (f.y ?? height / 2),
    }))
  return (
    <div className="editors">
      <Card title="Review controls" eyebrow="Human corrections">
        <label className="field">
          Reviewer name
          <input
            value={p.reviewer}
            onChange={(e) => p.setReviewer(e.target.value)}
            maxLength={100}
            placeholder="Needed to save, not to preview"
            autoComplete="name"
          />
        </label>
        <label className="field">
          Video interaction
          <select
            aria-label="Video interaction"
            value={p.tool}
            onChange={(e) => p.setTool(e.target.value as EditTool)}
          >
            <option value="inspect">Inspect / playback</option>
            <option value="waterline">Click: waterline</option>
            <option value="roi">Drag: ROI</option>
            <option value="point">Click: tracking point</option>
            <option value="box">Drag: detector box</option>
            <option value="keypoint">Click: selected keypoint</option>
          </select>
        </label>
        <p className="small muted">
          Changes preview instantly in purple on the video and timeline. Draw on
          the video, nudge with arrow keys, or edit the fields; nothing is saved
          until you apply it. Changing frames discards unsaved frame drafts.
        </p>
      </Card>
      <Card
        title="Scene settings"
        eyebrow={
          p.edits.scene ? 'Manual correction saved' : 'Automatic baseline'
        }
      >
        <NumberField
          label="Waterline y (px)"
          value={scene.waterline ?? NaN}
          max={height}
          onChange={(n) => p.setScene((s) => ({ ...s, waterline: n }))}
        />
        <button onClick={() => p.setScene((s) => ({ ...s, waterline: null }))}>Clear waterline draft</button>
        <BoxFields
          box={scene.roi}
          set={(roi) => p.setScene((s) => ({ ...s, roi }))}
          prefix="ROI"
          width={width}
          height={height}
        />
        <p className="baseline">
          Automatic: waterline {session.scene.waterline ?? 'unavailable'} · ROI [
          {session.scene.roi.join(', ')}]
        </p>
        <div className="button-row">
          <button className="primary" onClick={p.saveScene}>
            Apply scene
          </button>
          <button onClick={p.resetScene}>
            <Icon name="reset" />
            Restore automatic
          </button>
        </div>
      </Card>
      {session.overlay.t.length > 0 && (
        <Card
          title={`Frame ${p.frameIndex} correction`}
          eyebrow="Tracking point + detector output"
        >
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={!frame.detected}
              onChange={(e) => markMissing(e.target.checked)}
            />
            Fish missing in this frame
          </label>
          <div className="field-grid">
            <NumberField
              label="Track x (px)"
              value={frame.x ?? NaN}
              max={width}
              onChange={(n) =>
                p.setFrame((f) => ({ ...f, x: n, detected: true }))
              }
            />
            <NumberField
              label="Track y (px)"
              value={frame.y ?? NaN}
              max={height}
              onChange={(n) =>
                p.setFrame((f) => ({ ...f, y: n, detected: true }))
              }
            />
          </div>
          {frame.box ? (
            <BoxFields
              box={frame.box}
              set={(box) => p.setFrame((f) => ({ ...f, box }))}
              prefix="Box"
              width={width}
              height={height}
            />
          ) : (
            <p className="baseline">No detector box for this frame.</p>
          )}
          {kp ? (
            <>
              <label className="field">
                Keypoint
                <select
                  aria-label="Keypoint"
                  value={p.keypoint}
                  onChange={(e) => p.setKeypoint(e.target.value)}
                >
                  {Object.keys(frame.keypoints).map((k) => (
                    <option key={k} value={k}>
                      {k.replaceAll('_', ' ')}
                    </option>
                  ))}
                </select>
              </label>
              <div className="field-grid">
                <NumberField
                  label="Keypoint x (px)"
                  value={kp[0]}
                  max={width}
                  onChange={(n) =>
                    p.setFrame((f) => ({
                      ...f,
                      keypoints: {
                        ...f.keypoints,
                        [p.keypoint]: [n, kp[1], kp[2]],
                      },
                    }))
                  }
                />
                <NumberField
                  label="Keypoint y (px)"
                  value={kp[1]}
                  max={height}
                  onChange={(n) =>
                    p.setFrame((f) => ({
                      ...f,
                      keypoints: {
                        ...f.keypoints,
                        [p.keypoint]: [kp[0], n, kp[2]],
                      },
                    }))
                  }
                />
              </div>
              <p className="baseline">
                Keypoint confidence: {Math.round(kp[2] * 100)}% · read-only
              </p>
            </>
          ) : (
            <p className="baseline">No keypoints for this frame.</p>
          )}
          <details className="original-frame">
            <summary>Compare with automatic frame</summary>
            <pre>
              {JSON.stringify(
                {
                  ...frameAt(session, p.frameIndex, emptyEdits()),
                  detection:
                    session.overlay.detections.find(
                      (d) => d.frame_idx === p.frameIndex,
                    ) ?? null,
                },
                null,
                2,
              )}
            </pre>
          </details>
          <div className="button-row">
            <button className="primary" onClick={p.saveFrame}>
              Apply frame
            </button>
            <button onClick={p.resetFrame}>Restore frame</button>
          </div>
        </Card>
      )}
    </div>
  )
}

export function LabelEditor(
  p: Pick<
    EditorProps,
    | 'session'
    | 'edits'
    | 'range'
    | 'setRange'
    | 'state'
    | 'setState'
    | 'saveLabels'
    | 'resetLabels'
  > & {
    pending: LabelEdit[]
    stage: () => void
    saveStaged: () => void
    removeStaged: (index: number) => void
    discardStaged: () => void
  },
) {
  const { session } = p
  return (
    <Card title="Relabel a time range" eyebrow="Behavior override">
      <div className="field-grid">
        <NumberField
          label="From (s)"
          value={p.range[0]}
          max={session.duration}
          onChange={(n) => p.setRange([n, p.range[1]])}
        />
        <NumberField
          label="To (s)"
          value={p.range[1]}
          max={session.duration}
          onChange={(n) => p.setRange([p.range[0], n])}
        />
      </div>
      <label className="field">
        Behavior state
        <select
          aria-label="Behavior state"
          value={p.state}
          onChange={(e) => p.setState(e.target.value as Behavior)}
        >
          {STATES.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </label>
      <div className="button-row">
        <button className="primary" onClick={p.saveLabels} disabled={p.pending.length > 0}>
          Apply behavior
        </button>
        <button onClick={p.stage}>Stage range</button>
        <button onClick={p.resetLabels}>Restore labels</button>
      </div>
      <p className="small muted">
        Latest overlapping correction wins. Automatic labels remain visible in
        the timeline.
      </p>
      {p.pending.length > 0 && <>
        <h3>Staged edits · not saved</h3>
        <ul className="edit-list">
          {p.pending.map((edit, i) => <li key={i}>
            <b>{edit.start_s.toFixed(2)}–{edit.end_s.toFixed(2)} s</b> · {edit.state}
            <button aria-label={`Remove staged edit ${i + 1}`} onClick={() => p.removeStaged(i)}>Remove</button>
          </li>)}
        </ul>
        <div className="button-row">
          <button className="primary" onClick={p.saveStaged}>Save staged edits</button>
          <button onClick={p.discardStaged}>Discard staged edits</button>
        </div>
      </>}
      {p.edits.labels.length > 0 && (
        <ul className="edit-list">
          {p.edits.labels.map((e, i) => (
            <li key={i}>
              <b>
                {e.start_s.toFixed(2)}–{e.end_s.toFixed(2)} s
              </b>{' '}
              · {e.state}
              <span>by {e.reviewer}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
