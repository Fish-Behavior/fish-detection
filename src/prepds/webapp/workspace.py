"""File-backed React workspace. Automatic pipeline artifacts remain the reset baseline."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import pandas as pd
import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prepds import review_store
from prepds.calibration.profile import labeling_thresholds
from prepds.config import ConfigError, Settings
from prepds.consolidate import consolidate_states
from prepds.features import compute_feature_validity, derive_features
from prepds.labeling import classify_video
from prepds.models import BehaviorState, FrameSource, MatchStatus, StateFrame, Track, Trial
from prepds.pipeline import Outcome, RunConfig, process_trial, video_id
from prepds.segments import DeadMonotonicityError, frames_to_segments
from prepds.video_io import probe
from prepds.webapp.schemas import Reviewer

LOG = logging.getLogger(__name__)
Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Box = tuple[Finite, Finite, Finite, Finite]
Keypoint = tuple[Finite, Finite, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]]


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Scene(Input):
    roi: Box
    waterline: Finite | None
    reviewer: Reviewer


class Frame(Input):
    x: Finite | None
    y: Finite | None
    detected: bool = Field(strict=True)
    box: Box | None
    keypoints: dict[str, Keypoint] = Field(max_length=20)
    reviewer: Reviewer


class Label(Input):
    start_s: Finite
    end_s: Finite
    state: BehaviorState
    reviewer: Reviewer


class Final(Input):
    compound: Annotated[str, Field(min_length=1, max_length=100)]
    dose: Annotated[str, Field(max_length=100)]
    reviewer: Reviewer


class Edits(Input):
    scene: Scene | None = None
    frames: dict[str, Frame] = Field(default_factory=dict, max_length=10000)
    labels: list[Label] = Field(default_factory=list, max_length=1000)
    finalResult: Final | None = None


class Revision(Input):
    revision: int = Field(ge=0, strict=True)
    baseline: str
    reviewer: Reviewer


class ReviewIn(Revision):
    edits: Edits
    decision: Literal['ACCEPTED', 'REJECTED'] | None = None


class ProcessIn(Input):
    reviewer: Reviewer


class Message(Input):
    role: Literal['user', 'assistant']
    content: str = Field(max_length=20000)


class Question(Input):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=100)


class Answer(BaseModel):
    answer: str
    tool_calls: list
    history: list[Message]


def fail(status: int, code: str, message: str):
    raise HTTPException(status, {'code': code, 'message': message})


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.workspace.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path: Path, default: dict):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        fail(500, 'corrupt_artifact', f'{path.name} is unreadable; restore or regenerate it.')


def tracks_of(frames: pd.DataFrame, edits: Edits | None = None):
    tracks = []
    for r in frames.itertuples(index=False):
        correction = edits.frames.get(str(r.frame_idx)) if edits else None
        x, y, detected = (correction.x, correction.y, correction.detected) if correction else (r.x, r.y, r.detected)
        if edits and edits.scene and detected:
            x0, y0, x1, y1 = edits.scene.roi
            detected = x0 <= x <= x1 and y0 <= y <= y1
        orientation = float(r.orientation_deg) if detected and pd.notna(r.orientation_deg) else None
        # Manual centroid/box/keypoint changes do not establish a calibrated body angle.
        if correction:
            orientation = None
        tracks.append(Track(int(r.frame_idx), float(r.t_sec), float(x) if detected else 0.,
                            float(y) if detected else 0., orientation, float(y) if detected else None, bool(detected)))
    return tracks


def measurements_of(tracks, waterline=None):
    values, valid = derive_features(tracks), compute_feature_validity(tracks)
    result, last_second = [], -1
    for i, (track, feature) in enumerate(zip(tracks, values)):
        second = math.floor(track.t_sec)
        if second == last_second:
            continue
        last_second = second
        result.append({'t': track.t_sec, 'speed': feature.velocity if valid.velocity[i] else None,
                       'turning': feature.angular_velocity if valid.angular_velocity[i] else None,
                       'depth': track.y - waterline if track.detected and waterline is not None else None})
    return result


def install_workspace(app, processed_dir: Path, *, video_dir: Path | None, profile_dir: Path | None,
                      settings: Settings | None, chat_url: str | None, predictions_path: Path | None,
                      classifier_root: Path | None, model_run: Path | None,
                      overlay_window, video_detail, explain_second, clock):
    from prepds.webapp.app import _ID, _VIDEO_SUFFIXES, _resolve_source

    router = APIRouter(prefix='/api')
    active: set[str] = set()
    active_lock = threading.Lock()
    process_slot = threading.Semaphore(1)
    if chat_url and (urlparse(chat_url).scheme != 'http' or urlparse(chat_url).hostname not in ('localhost', '127.0.0.1', '::1')
                     or urlparse(chat_url).username or urlparse(chat_url).password or urlparse(chat_url).query or urlparse(chat_url).fragment):
        raise ConfigError('chat URL must be an HTTP service on loopback, without credentials, query or fragment')

    def catalog():
        path = processed_dir.parent / 'trials_catalog.parquet'
        if not path.is_file():
            return []
        raw = pd.read_parquet(path).astype(object)
        return [Trial.from_dict(r) for r in raw.where(raw.notna(), None).to_dict('records')]

    def inventory():
        records = {}
        by_source = {}
        for trial in catalog():
            if trial.video_path is not None and trial.match_status == MatchStatus.MATCHED:
                source = _resolve_source(trial.video_path, video_dir)
                vid = video_id(trial)
                if source and _ID.fullmatch(vid):
                    if source in by_source or vid in records:
                        fail(409, 'duplicate_video', 'The catalog has duplicate video identities; fix the catalog before processing.')
                    records[vid] = {'source': source, 'trial': trial, 'processed': False}
                    by_source[source] = vid
        if processed_dir.is_dir():
            for directory in sorted(processed_dir.iterdir()):
                if not directory.is_dir() or not _ID.fullmatch(directory.name) or directory.is_symlink() or not (directory / 'manifest.json').is_file():
                    continue
                try:
                    manifest = review_store.load_manifest(directory)
                except (OSError, ValueError, KeyError, TypeError):
                    records[directory.name] = {'source': None, 'processed': True, 'error': 'Manifest is unreadable.'}
                    continue
                source = _resolve_source(manifest.video_path, video_dir)
                if source and source in by_source and by_source[source] != directory.name:
                    fail(409, 'duplicate_video', 'Multiple pipeline identities refer to one source video.')
                records.setdefault(directory.name, {})
                records[directory.name].update(source=source, processed=True, manifest=manifest)
                if source:
                    by_source[source] = directory.name
        if video_dir and video_dir.is_dir():
            for path in sorted(video_dir.rglob('*')):
                if path.suffix.lower() not in _VIDEO_SUFFIXES:
                    continue
                source = _resolve_source(path, video_dir)
                if not source or source in by_source:
                    continue
                relative = source.relative_to(video_dir.resolve()).as_posix()
                vid = 'raw_' + hashlib.sha256(relative.encode()).hexdigest()[:20]
                records[vid] = {'source': source, 'processed': False}
                by_source[source] = vid
        return records

    def record(vid):
        if not _ID.fullmatch(vid):
            fail(404, 'unknown_video', 'Unknown video.')
        r = inventory().get(vid)
        if not r:
            fail(404, 'unknown_video', 'Unknown video.')
        if r.get('error'):
            fail(500, 'corrupt_manifest', r['error'])
        return r

    def directory(vid):
        path = processed_dir / vid
        if not path.resolve().is_relative_to(processed_dir.resolve()) or path.is_symlink():
            fail(404, 'unknown_video', 'Unknown video.')
        return path

    def baseline(vid, r):
        paths = [r['source']] if r['source'] else []
        paths += [directory(vid) / name for name in ('manifest.json', 'frames.parquet', 'detections.parquet', 'segments.csv', 'waterline.json')]
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()]
        return hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    def review(vid, r):
        default = {'revision': 0, 'baseline': baseline(vid, r), 'edits': Edits().model_dump(mode='json'),
                   'decision': None, 'reviewer': None, 'updated_at': None, 'derived': None}
        value = read_json(directory(vid) / 'review.json', default)
        try:
            Edits.model_validate(value['edits'])
            if not isinstance(value['revision'], int) or value['revision'] < 0:
                raise ValueError()
            if value['baseline'] != baseline(vid, r):
                fail(409, 'baseline_changed', 'The source or automatic outputs changed. Restore the matching files or archive review.json before reviewing the new baseline.')
        except (ValueError, KeyError, TypeError):
            fail(500, 'corrupt_review', 'Saved review is unreadable; restore review.json.')
        return value

    def job(vid):
        value = read_json(directory(vid) / 'processing.json', {'status': 'idle', 'message': ''})
        if value['status'] in ('queued', 'running') and vid not in active:
            return {**value, 'status': 'interrupted', 'message': 'Processing was interrupted; retry to regenerate.'}
        return value

    def asset(r):
        if r['source'] is None:
            fail(404, 'source_missing', 'Source video is missing or outside the configured video folder.')
        try:
            return probe(r['source'])
        except (OSError, ValueError):
            fail(409, 'unreadable_video', 'Source video cannot be decoded. Replace or convert it to a supported recording.')

    def predictions(vid, path=None):
        path = path or predictions_path
        if not path or not path.is_file():
            return None, None
        try:
            table = pd.read_csv(path) if path.suffix == '.csv' else pd.read_parquet(path)
            rows = table.loc[table['video_id'] == vid]
            if rows.empty:
                return None, None
            if len(rows) != 1:
                raise ValueError()
            row = rows.iloc[0]
            probs = {k: float(row[k]) for k in table.columns if k.startswith('p:')}
            if not probs or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()) or 'p:' + str(row['predicted']) not in probs:
                raise ValueError()
            return {'video_id': vid, 'predicted': str(row['predicted']), **probs}, None
        except (OSError, ValueError, KeyError, TypeError):
            return None, 'Model predictions are unreadable or invalid.'

    def stale(value):
        edits = Edits.model_validate(value['edits'])
        upstream = edits.scene is not None or bool(edits.frames)
        changed = upstream or bool(edits.labels)
        calculated = value.get('derived') is not None
        return {'measurements': changed and not calculated, 'labels': upstream and not calculated,
                'predictions': changed and value.get('model_result') is None}

    def frames_of(vid):
        path = directory(vid) / 'frames.parquet'
        if not path.is_file():
            fail(409, 'missing_artifacts', 'No tracking results. Process the video first.')
        return pd.read_parquet(path).sort_values('frame_idx')

    def segments_of(frames, edits=None):
        states = []
        for r in frames.itertuples(index=False):
            manual = next((e for e in reversed(edits.labels) if e.start_s <= r.t_sec < e.end_s), None) if edits else None
            states.append(StateFrame(int(r.frame_idx), float(r.t_sec), manual.state if manual else BehaviorState(r.state),
                                     FrameSource.MANUAL if manual else FrameSource(r.source), None))
        return [s.to_dict() for s in frames_to_segments(states)]

    @router.get('/health')
    def health():
        return {'ok': True, 'persistence': 'files', 'chat_configured': bool(chat_url), 'processing_configured': settings is not None,
                'predictions_configured': predictions_path is not None, 'inference_configured': bool(classifier_root and model_run),
                'instance_id': os.environ.get('FISHLAB_INSTANCE_ID')}

    @router.get('/sessions')
    def sessions():
        rows = []
        for vid, r in sorted(inventory().items()):
            state = job(vid)
            status = 'error' if r.get('error') else state['status'] if state['status'] != 'idle' else 'processed' if r['processed'] else 'unprocessed'
            value = read_json(directory(vid) / 'review.json', {})
            e = value.get('edits', {})
            edited = any(e.get(key) for key in ('scene', 'frames', 'labels', 'finalResult'))
            rows.append({'video_id': vid, 'name': r['source'].name if r['source'] else vid, 'processing_status': status,
                         'processing_message': r.get('error') or state['message'], 'processable': bool(r.get('trial')),
                         'review_status': value.get('decision') or ('EDITED' if edited else 'PROCESSED_AUTO' if r['processed'] else 'NOT_PROCESSED'),
                         'edited': edited})
        return rows

    @router.get('/sessions/{vid}/video')
    def video(vid: str):
        from fastapi.responses import FileResponse
        r = record(vid)
        if not r['source']:
            fail(404, 'source_missing', 'Source video not found.')
        return FileResponse(r['source'], media_type='video/mp4' if r['source'].suffix.lower() in ('.mp4', '.m4v') else None)

    @router.get('/sessions/{vid}/overlay')
    def overlay(vid: str, start_s: float = Query(ge=0, allow_inf_nan=False), end_s: float = Query(gt=0, allow_inf_nan=False)):
        if not 0 < end_s - start_s <= 30:
            fail(422, 'invalid_window', 'Overlay windows must span more than 0 and at most 30 seconds.')
        r = record(vid); a = asset(r)
        empty = {'fps': a.fps, 'width': a.resolution[0], 'height': a.resolution[1], 't': [], 'frame_idx': [],
                 'x': [], 'y': [], 'detected': [], 'detections': [], 'detections_error': None, 'has_detector': False}
        if not r['processed']:
            return empty
        body = overlay_window(vid, start_s, end_s).model_dump(mode='json')
        indices = pd.read_parquet(directory(vid) / 'frames.parquet', columns=['frame_idx', 't_sec'],
                                  filters=[('t_sec', '>=', start_s), ('t_sec', '<', end_s)])
        return {**body, 'width': a.resolution[0], 'height': a.resolution[1], 'frame_idx': indices.frame_idx.astype(int).tolist()}

    def read_original(vid, expected_baseline, reader):
        r = record(vid)
        signature = review(vid, r)['baseline']
        if expected_baseline is not None and signature != expected_baseline:
            fail(409, 'baseline_changed', 'The pipeline outputs changed. Reload before inspecting the original labels.')
        value = reader()
        if baseline(vid, r) != signature:
            fail(409, 'baseline_changed', 'The pipeline outputs changed while loading. Reload before inspecting them.')
        return value

    @router.get('/sessions/{vid}/review-tools')
    def review_tools(vid: str, expected_baseline: str | None = Query(None, alias='baseline')):
        data = read_original(vid, expected_baseline, lambda: video_detail(vid))
        return {
            'manifest': data.manifest.model_dump(mode='json', exclude={'video_path'}),
            'listing_flags': [flag.model_dump() for flag in data.listing_flags],
            'listing_flags_error': data.listing_flags_error,
        }

    @router.get('/sessions/{vid}/explain')
    def explanation(vid: str, second: int = Query(ge=0), expected_baseline: str | None = Query(None, alias='baseline')):
        return read_original(vid, expected_baseline, lambda: explain_second(vid, second))

    @router.get('/sessions/{vid}')
    def session(vid: str):
        r = record(vid); a = asset(r); value = review(vid, r)
        frames = frames_of(vid) if r['processed'] else None
        scene = {'roi': [0, 0, *a.resolution], 'waterline': None}
        waterline = read_json(directory(vid) / 'waterline.json', {})
        if 'y_px' in waterline and math.isfinite(float(waterline['y_px'])) and 0 <= waterline['y_px'] <= a.resolution[1]:
            scene['waterline'] = waterline['y_px']
        derived = value.get('derived')
        prediction, prediction_error = predictions(vid)
        model_result = value.get('model_result')
        if model_result:
            prediction = model_result['prediction']
        warnings = []
        if r['processed'] and abs(r['manifest'].video_duration_s - a.duration_s) > 1 / a.fps:
            warnings.append('Source duration differs from the pipeline output. Verify the source recording before editing.')
        if prediction_error:
            warnings.append(prediction_error)
        return {'video_id': vid, 'name': a.path.name, 'video_url': f'/api/sessions/{vid}/video', 'duration': a.duration_s,
                'frame_count': a.frame_count, 'scene': scene, 'overlay': overlay(vid, 0, min(30, a.duration_s)),
                'segments': segments_of(frames) if frames is not None else [],
                'reviewed_segments': derived['segments'] if derived else None,
                'measurements': derived['measurements'] if derived else measurements_of(tracks_of(frames), scene['waterline']) if frames is not None else [],
                'predictions': prediction, 'review': {k: v for k, v in value.items() if k != 'derived'}, 'baseline': value['baseline'],
                'stale': stale(value), 'warnings': warnings, 'processing': job(vid), 'processable': bool(r.get('trial')),
                'has_tracking': frames is not None, 'chat_available': bool(chat_url),
                'model_provenance': model_result and {'run': model_result['run'], 'baseline': model_result['baseline']}}

    def check_revision(vid, r, body):
        current = review(vid, r)
        if current['revision'] != body.revision or current['baseline'] != body.baseline:
            fail(409, 'revision_conflict', 'Another tab saved changes. Reload the session before saving again.')
        if job(vid)['status'] in ('queued', 'running'):
            fail(409, 'processing_busy', 'Wait for processing to finish before saving.')
        return current

    def validate(vid, r, edits):
        a = asset(r); width, height = a.resolution
        def point(x, y):
            if x is None or y is None or not 0 <= x <= width or not 0 <= y <= height:
                fail(422, 'invalid_coordinates', f'Coordinates must lie inside the {width} × {height} frame.')
        def box(b):
            point(b[0], b[1]); point(b[2], b[3])
            if b[0] >= b[2] or b[1] >= b[3]:
                fail(422, 'invalid_box', 'Box edges must have positive width and height.')
        if edits.scene:
            box(edits.scene.roi)
            if edits.scene.waterline is not None and not 0 <= edits.scene.waterline <= height:
                fail(422, 'invalid_waterline', 'Waterline must lie inside the frame.')
        for index, f in edits.frames.items():
            if not index.isdecimal() or str(int(index)) != index or not 0 <= int(index) < a.frame_count:
                fail(422, 'invalid_frame', 'Frame index is outside the recording.')
            if f.detected:
                point(f.x, f.y)
            elif f.x is not None or f.y is not None:
                fail(422, 'invalid_frame', 'Missing fish must have null coordinates.')
            if f.box:
                box(f.box)
            for name, kp in f.keypoints.items():
                if name not in ('snout', 'dorsal_fin_base', 'ventral', 'tail_base', 'tail_tip'):
                    fail(422, 'invalid_keypoint', 'Unknown detector keypoint.')
                point(kp[0], kp[1])
        for e in edits.labels:
            if not 0 <= e.start_s < e.end_s <= a.duration_s:
                fail(422, 'invalid_interval', 'Behavior intervals must be within the recording, with start before end.')
        if edits.finalResult and not edits.finalResult.compound.strip():
            fail(422, 'invalid_result', 'Final compound must not be blank.')
        if edits.labels:
            try:
                segments_of(frames_of(vid), edits)
            except DeadMonotonicityError:
                fail(422, 'invalid_labels', 'Dead must remain terminal through the end of the recording.')

    @router.put('/sessions/{vid}/review')
    def save_review(vid: str, body: ReviewIn):
        r = record(vid)
        with locked(directory(vid)):
            current = check_revision(vid, r, body)
            validate(vid, r, body.edits)
            edits = body.edits.model_dump(mode='json')
            same_inputs = all(edits[key] == current['edits'][key] for key in ('scene', 'frames', 'labels'))
            value = {**current, 'revision': current['revision'] + 1, 'edits': edits, 'decision': body.decision,
                     'reviewer': body.reviewer, 'updated_at': clock().isoformat(),
                     'derived': current.get('derived') if same_inputs else None,
                     'model_result': current.get('model_result') if same_inputs else None}
            atomic_json(directory(vid) / 'review.json', value)
        return {k: v for k, v in value.items() if k != 'derived'}

    @router.post('/sessions/{vid}/rerun')
    def rerun(vid: str, body: Revision):
        r = record(vid)
        with locked(directory(vid)):
            current = check_revision(vid, r, body)
            edits = Edits.model_validate(current['edits']); frames = frames_of(vid)
            tracks = tracks_of(frames, edits)
            if edits.scene or edits.frames:
                version = r['manifest'].calibration_profile_version
                path = profile_dir / (version + '.yaml') if profile_dir and _ID.fullmatch(version) else None
                if not path or not path.is_file():
                    fail(503, 'calibration_missing', 'The exact calibration profile used by this video is unavailable.')
                thresholds = labeling_thresholds(yaml.safe_load(path.read_text(encoding='utf-8')))
                states = consolidate_states(classify_video(tracks, **thresholds))
                frames = frames.copy()
                frames['state'] = [s.state.value for s in states]
                frames['source'] = FrameSource.AUTO.value
            try:
                segments = segments_of(frames, edits)
            except DeadMonotonicityError:
                fail(422, 'invalid_labels', 'Corrections would make Dead nonterminal; revise the labels.')
            scene = edits.scene
            # Waterline is for surface-relative display only; calibrated labels use frame-top y.
            waterline = scene.waterline if scene else read_json(directory(vid) / 'waterline.json', {}).get('y_px')
            current.update(revision=current['revision'] + 1, reviewer=body.reviewer, updated_at=clock().isoformat(),
                           derived={'segments': segments, 'measurements': measurements_of(tracks, waterline)}, model_result=None)
            atomic_json(directory(vid) / 'review.json', current)
        return session(vid)

    @router.post('/sessions/{vid}/predict')
    def predict(vid: str, body: Revision):
        r = record(vid)
        if not classifier_root or not model_run or not (classifier_root / 'src/dcs').is_dir() or not (model_run / 'model/model_info.json').is_file():
            fail(503, 'model_unavailable', 'Set --classifier-root to the DCS checkout and --model-run to a trained DCS run.')
        with locked(directory(vid)):
            current = check_revision(vid, r, body)
            if stale(current)['measurements'] or stale(current)['labels']:
                fail(409, 'stale_features', 'Rerun affected stages before compound inference.')
            edits = Edits.model_validate(current['edits'])
            frames = frames_of(vid).copy()
            tracks = tracks_of(frames, edits)
            for i, (track, feature) in enumerate(zip(tracks, derive_features(tracks))):
                for key in ('x', 'y', 'orientation_deg', 'detected'):
                    frames.loc[frames.index[i], key] = getattr(track, key)
                # DCS depth features use the existing frame-top contract, never manual waterline depth.
                frames.loc[frames.index[i], 'depth_from_surface'] = track.y_from_frame_top
                for key in ('velocity', 'acceleration', 'angular_velocity', 'meander', 'is_immobile'):
                    frames.loc[frames.index[i], key] = getattr(feature, key)
            segments = current['derived']['segments'] if current.get('derived') else segments_of(frames, edits)
            frames['state'] = frames['state'].astype(str)
            frames['source'] = frames['source'].astype(str)
            for segment in segments:
                mask = (frames.t_sec >= segment['start_s']) & (frames.t_sec < segment['end_s'])
                frames.loc[mask, 'state'] = segment['state']
                frames.loc[mask, 'source'] = segment['source']
            frames['state'] = pd.Categorical(frames['state'], categories=[s.value for s in BehaviorState])
            frames['source'] = pd.Categorical(frames['source'], categories=[s.value for s in FrameSource])
            # DCS reads its own contract and preprocessing; no classifier math is copied into prepds.
            with tempfile.TemporaryDirectory(prefix='fishlab-inference-') as temp:
                root = Path(temp); snapshot = root / 'videos' / vid; snapshot.mkdir(parents=True)
                frames.to_parquet(snapshot / 'frames.parquet', index=False)
                pd.DataFrame(segments).to_csv(snapshot / 'segments.csv', index=False)
                shutil.copy2(directory(vid) / 'manifest.json', snapshot / 'manifest.json')
                if (directory(vid) / 'detections.parquet').is_file():
                    shutil.copy2(directory(vid) / 'detections.parquet', snapshot / 'detections.parquet')
                table, output = root / 'table.parquet', root / 'predictions.csv'
                commands = [
                    [sys.executable, '-m', 'dcs', 'featurize', '--videos', str(root / 'videos'), '--out', str(table)],
                    [sys.executable, '-m', 'dcs', 'predict', '--model', str(model_run.resolve()), '--input', str(table), '--out', str(output)],
                ]
                env = {**os.environ, 'PYTHONPATH': str(classifier_root.resolve() / 'src')}
                # Exported paths belong to the app's cwd, before DCS changes checkout.
                for key in ('DCS_ACCEPTED_DIR', 'DCS_DB_PATH', 'DCS_PROCESSED_DIR', 'DCS_OUTPUT_DIR', 'DCS_TABLE', 'DCS_CONFIG'):
                    if env.get(key, '').strip():
                        env[key] = str(Path(env[key].strip()).expanduser().resolve())
                try:
                    for command in commands:
                        result = subprocess.run(command, cwd=classifier_root, env=env, capture_output=True, text=True, timeout=120)
                        if result.returncode:
                            LOG.error('DCS inference failed: %s %s', result.stdout, result.stderr)
                            fail(503, 'inference_failed', 'DCS could not featurize or predict this recording. Check model/profile/workbook compatibility in the backend log.')
                    prediction, error = predictions(vid, output)
                    if prediction is None:
                        fail(503, 'inference_failed', error or 'DCS dropped this recording; check its feature requirements.')
                except (OSError, subprocess.TimeoutExpired):
                    LOG.exception('DCS inference failed')
                    fail(503, 'inference_failed', 'DCS inference could not finish. Check the configured DCS checkout and dependencies.')
            current.update(revision=current['revision'] + 1, reviewer=body.reviewer, updated_at=clock().isoformat(),
                           model_result={'prediction': prediction, 'run': model_run.name, 'baseline': current['baseline']})
            atomic_json(directory(vid) / 'review.json', current)
        return session(vid)

    @router.get('/sessions/{vid}/processing')
    def processing_status(vid: str):
        record(vid)
        return job(vid)

    def run_process(vid, trial, config, reviewer):
        path = directory(vid) / 'processing.json'
        try:
            with process_slot:
                atomic_json(path, {'status': 'running', 'message': '', 'reviewer': reviewer})
                result = process_trial(trial, config)
            if result.outcome != Outcome.PROCESSED:
                LOG.error('Processing %s: %s', vid, result.message)
            atomic_json(path, {'status': 'processed' if result.outcome == Outcome.PROCESSED else 'failed',
                               'message': '' if result.outcome == Outcome.PROCESSED else 'Analysis failed. Check the backend log and retry.',
                               'reviewer': reviewer, 'updated_at': clock().isoformat()})
        except Exception:
            LOG.exception('Processing %s failed', vid)
            atomic_json(path, {'status': 'failed', 'message': 'Analysis failed. Check the backend log and retry.'})
        finally:
            with active_lock:
                active.discard(vid)

    @router.post('/sessions/{vid}/process', status_code=202)
    def process(vid: str, body: ProcessIn, tasks: BackgroundTasks):
        r = record(vid)
        if settings is None:
            fail(503, 'processing_unconfigured', 'Start prepds review with pipeline settings to enable analysis.')
        if not r.get('trial'):
            fail(409, 'unmatched_video', 'This recording has no unique matched trial. Run prepds catalog and resolve the workbook match.')
        asset(r)
        with locked(directory(vid)):
            saved = read_json(directory(vid) / 'review.json', {})
            if r['processed'] and r['manifest'].review_status.value != 'REJECTED' or saved:
                fail(409, 'reviewed_work', 'Existing output or review corrections must not be overwritten. Use Rerun for corrections.')
            with active_lock:
                if vid in active:
                    fail(409, 'processing_busy', 'This recording is already being processed.')
                active.add(vid)
            try:
                from prepds import __version__
                config = RunConfig(processed_dir, labeling_thresholds(settings.params), __version__,
                                   str(settings.params['calibration_profile']['version']), clock())
                value = {'status': 'queued', 'message': '', 'reviewer': body.reviewer}
                atomic_json(directory(vid) / 'processing.json', value)
            except Exception:
                active.discard(vid)
                raise
            tasks.add_task(run_process, vid, replace(r['trial'], video_path=r['source']), config, body.reviewer)
        return value

    @router.post('/ask', response_model=Answer)
    def ask(body: Question):
        if not body.question.strip():
            fail(422, 'empty_question', 'Enter a research question.')
        if not chat_url:
            fail(503, 'chat_unconfigured', 'Research chat is unavailable. Start the DCS chat service and set --chat-url.')
        request = urllib.request.Request(chat_url.rstrip('/') + '/api/ask',
                                         data=body.model_dump_json().encode(), headers={'Content-Type': 'application/json'}, method='POST')
        try:
            # Refuse redirects so the local data boundary cannot become a remote relay.
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args, **kwargs):
                    return None
            with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(request, timeout=90) as response:
                return Answer.model_validate(json.loads(response.read(2_000_000)))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, ValidationError):
            fail(503, 'chat_unavailable', 'Research chat failed or returned an invalid response. Check the DCS service and model.')

    app.include_router(router)
