"""React backend contract, persistence, trust boundaries and real recalculation."""
import json
import datetime as dt
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from prepds.config import load_settings
from prepds.webapp.app import create_app
from tests.review.test_webapp_api import _make


@pytest.fixture
def workspace(tmp_path):
    processed, sources = tmp_path / 'processed', tmp_path / 'videos'
    processed.mkdir(); sources.mkdir()
    vid = _make(processed, sources)
    import cv2
    capture = cv2.VideoCapture('tests/fixtures/synth_tiny.mp4')
    ok, frame = capture.read(); capture.release()
    assert ok
    writer = cv2.VideoWriter(str(sources / f'{vid}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (304, 240))
    for _ in range(300):
        writer.write(frame)
    writer.release()
    def client():
        return TestClient(create_app(processed, tmp_path / 'accepted', video_dir=sources,
                                    profile_dir=Path('config/calibration_profiles'),
                                    allowed_hosts=('testserver',)))
    return client, processed / vid, sources, vid


def test_session_and_bounded_overlay(workspace):
    make_client, _, _, vid = workspace
    client = make_client()
    rows = client.get('/api/sessions').json()
    assert rows[0]['video_id'] == vid and rows[0]['processing_status'] == 'processed'
    r = client.get(f'/api/sessions/{vid}')
    assert r.status_code == 200
    s = r.json()
    assert s['overlay']['width'] > 0 and s['overlay']['frame_idx'][0] == 0
    assert s['predictions'] is None and s['scene']['waterline'] is None
    assert s['review']['revision'] == 0
    assert client.get(f'/api/sessions/{vid}/overlay', params={'start_s': 0, 'end_s': 31}).status_code == 422


def test_repository_relative_manifest_source_loads_without_rewriting_it(workspace, monkeypatch):
    make_client, directory, sources, vid = workspace
    nested = sources / 'nested'; nested.mkdir()
    (sources / f'{vid}.mp4').rename(nested / f'{vid}.mp4')
    manifest = json.loads((directory / 'manifest.json').read_text())
    monkeypatch.chdir(sources.parent)
    manifest['video_path'] = f'videos/nested/{vid}.mp4'
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    before = (directory / 'manifest.json').read_bytes()
    client = make_client()
    assert client.get(f'/api/sessions/{vid}').status_code == 200
    assert client.get(f'/videos/{vid}/video').status_code == 200
    rows = client.get('/api/sessions').json()
    assert [r['video_id'] for r in rows] == [vid]
    assert (directory / 'manifest.json').read_bytes() == before
    duplicate = sources / 'duplicate'; duplicate.mkdir()
    shutil.copy(nested / f'{vid}.mp4', duplicate / f'{vid}.mp4')
    assert client.get(f'/api/sessions/{vid}').status_code == 200
    assert client.get(f'/videos/{vid}/video').status_code == 200


def test_repository_relative_lookup_does_not_follow_symlinks_outside_sources(workspace, tmp_path, monkeypatch):
    make_client, directory, sources, vid = workspace
    outside = tmp_path / 'outside'; outside.mkdir()
    (sources / f'{vid}.mp4').rename(outside / f'{vid}.mp4')
    nested = sources / 'nested'; nested.mkdir()
    (nested / f'{vid}.mp4').symlink_to(outside / f'{vid}.mp4')
    manifest = json.loads((directory / 'manifest.json').read_text())
    monkeypatch.chdir(sources.parent)
    manifest['video_path'] = f'videos/nested/{vid}.mp4'
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    client = make_client()
    assert client.get(f'/api/sessions/{vid}').status_code == 404
    assert client.get(f'/videos/{vid}/video').status_code == 404


def test_frontend_review_tools_reuse_prepds_without_mutating_artifacts(workspace):
    from prepds.listing_flags import ListingFlag, write_listing_flags
    make_client, directory, _, vid = workspace
    write_listing_flags(directory, [ListingFlag(1.5, 7.5, 0.99, 2)], meta={})
    client = make_client()
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    legacy = client.get(f'/videos/{vid}').json()
    response = client.get(f'/api/sessions/{vid}/review-tools')
    assert response.status_code == 200
    tools = response.json()
    assert tools['manifest']['subject_id'] == legacy['manifest']['subject_id']
    assert 'video_path' not in tools['manifest']
    assert tools['listing_flags'] == legacy['listing_flags']
    assert tools['listing_flags_error'] is None
    explanation = client.get(f'/api/sessions/{vid}/explain', params={'second': 1})
    assert explanation.status_code == 200
    assert explanation.json() == client.get(f'/videos/{vid}/explain', params={'second': 1}).json()
    assert client.get(f'/api/sessions/{vid}/explain', params={'second': -1}).status_code == 422
    assert client.get('/api/sessions/unknown/review-tools').status_code == 404
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
    (directory / 'listing_flags.json').write_text('{bad')
    response = client.get(f'/api/sessions/{vid}/review-tools')
    assert response.status_code == 200
    assert response.json()['listing_flags'] == [] and response.json()['listing_flags_error']


def test_corrections_survive_restart_and_conflicts_do_not_overwrite(workspace):
    make_client, directory, _, vid = workspace
    client = make_client()
    s = client.get(f'/api/sessions/{vid}').json()
    baseline = (directory / 'frames.parquet').read_bytes()
    body = {'revision': 0, 'baseline': s['baseline'], 'reviewer': 'Reviewer',
            'edits': s['review']['edits'], 'decision': None}
    body['edits']['labels'] = [{'start_s': 0, 'end_s': 1, 'state': 'Listing/LORR', 'reviewer': 'Reviewer'}]
    r = client.put(f'/api/sessions/{vid}/review', json=body)
    assert r.status_code == 200
    assert r.json()['revision'] == 1
    assert client.put(f'/api/sessions/{vid}/review', json=body).status_code == 409
    reloaded = make_client().get(f'/api/sessions/{vid}').json()
    assert reloaded['review']['edits'] == body['edits']
    assert reloaded['stale']['predictions']
    assert (directory / 'frames.parquet').read_bytes() == baseline


def test_batch_relabels_validate_together_and_latest_overlap_wins(workspace):
    make_client, directory, _, vid = workspace
    client = make_client(); s = client.get(f'/api/sessions/{vid}').json()
    edits = s['review']['edits']
    edits['labels'] = [
        {'start_s': 0, 'end_s': 2, 'state': 'Listing/LORR', 'reviewer': 'R'},
        {'start_s': 1, 'end_s': 2, 'state': 'Dead', 'reviewer': 'R'},
    ]
    body = {'revision': 0, 'baseline': s['baseline'], 'reviewer': 'R', 'edits': edits, 'decision': None}
    assert client.put(f'/api/sessions/{vid}/review', json=body).status_code == 422
    assert not (directory / 'review.json').exists()
    edits['labels'][1]['state'] = 'Erratic Movement'
    saved = client.put(f'/api/sessions/{vid}/review', json=body)
    assert saved.status_code == 200 and saved.json()['revision'] == 1
    result = client.post(f'/api/sessions/{vid}/rerun', json={
        'revision': 1, 'baseline': s['baseline'], 'reviewer': 'R'})
    assert result.status_code == 200
    segments = result.json()['reviewed_segments']
    assert segments[0]['state'] == 'Listing/LORR' and segments[0]['end_s'] == pytest.approx(1)
    assert segments[1]['state'] == 'Erratic Movement' and segments[1]['start_s'] == pytest.approx(1)


def test_review_readers_refuse_another_pipeline_baseline(workspace):
    make_client, _, sources, vid = workspace
    client = make_client()
    signature = client.get(f'/api/sessions/{vid}').json()['baseline']
    readers = [(f'/api/sessions/{vid}/review-tools', {}),
               (f'/api/sessions/{vid}/explain', {'second': 1})]
    for path, params in readers:
        assert client.get(path, params={**params, 'baseline': signature}).status_code == 200
    source = sources / f'{vid}.mp4'
    source.write_bytes(source.read_bytes() + b'changed')
    for path, params in readers:
        response = client.get(path, params={**params, 'baseline': signature})
        assert response.status_code == 409 and response.json()['detail']['code'] == 'baseline_changed'


def test_explanation_detects_pipeline_changes_during_read(workspace, monkeypatch):
    from prepds.explain import VideoExplainer
    make_client, _, sources, vid = workspace
    client = make_client()
    signature = client.get(f'/api/sessions/{vid}').json()['baseline']
    original = VideoExplainer.explain
    def replace_source(self, second):
        value = original(self, second)
        source = sources / f'{vid}.mp4'
        source.write_bytes(source.read_bytes() + b'changed')
        return value
    monkeypatch.setattr(VideoExplainer, 'explain', replace_source)
    response = client.get(f'/api/sessions/{vid}/explain', params={'second': 1, 'baseline': signature})
    assert response.status_code == 409 and response.json()['detail']['code'] == 'baseline_changed'


@pytest.mark.parametrize('edit', [
    {'scene': {'roi': [0, 0, 99999, 240], 'waterline': 0, 'reviewer': 'R'}},
    {'frames': {'-1': {'x': 1, 'y': 1, 'detected': True, 'box': None, 'keypoints': {}, 'reviewer': 'R'}}},
    {'frames': {'0': {'x': 1, 'y': None, 'detected': True, 'box': None, 'keypoints': {}, 'reviewer': 'R'}}},
    {'labels': [{'start_s': 0, 'end_s': 99999, 'state': 'Dead', 'reviewer': 'R'}]},
])
def test_invalid_corrections_never_write(workspace, edit):
    make_client, directory, _, vid = workspace
    client = make_client(); s = client.get(f'/api/sessions/{vid}').json()
    edits = s['review']['edits']; edits.update(edit)
    r = client.put(f'/api/sessions/{vid}/review', json={
        'revision': 0, 'baseline': s['baseline'], 'reviewer': 'R', 'edits': edits, 'decision': None})
    assert r.status_code == 422
    assert not (directory / 'review.json').exists()


def test_unprocessed_inventory_missing_chat_and_safe_errors(workspace):
    make_client, directory, sources, vid = workspace
    shutil.copy('tests/fixtures/synth_tiny.mp4', sources / 'unmatched.mp4')
    client = make_client()
    rows = client.get('/api/sessions').json()
    raw = next(r for r in rows if r['name'] == 'unmatched.mp4')
    assert raw['processing_status'] == 'unprocessed'
    assert client.get('/api/sessions/' + raw['video_id']).status_code == 200
    assert client.post('/api/ask', json={'question': 'hello', 'history': []}).status_code == 503
    assert client.post('/api/sessions/' + raw['video_id'] + '/process', json={'reviewer': 'R'}).status_code == 503
    (directory / 'manifest.json').write_text('{')
    r = client.get(f'/api/sessions/{vid}')
    assert r.status_code == 500 and str(directory) not in r.text


def test_foreign_origin_and_path_escape_are_refused(workspace, tmp_path):
    make_client, directory, sources, vid = workspace
    outside = tmp_path / 'outside'; outside.mkdir()
    shutil.copy('tests/fixtures/synth_tiny.mp4', outside / 'secret.mp4')
    (sources / 'leak.mp4').symlink_to(outside / 'secret.mp4')
    client = make_client()
    assert all(r['name'] != 'leak.mp4' for r in client.get('/api/sessions').json())
    assert client.post('/api/ask', json={'question': 'q'}, headers={'origin': 'http://evil.example'}).status_code == 403
    assert client.get('/api/sessions/%2e%2e').status_code == 404


def test_rerun_uses_existing_calibration_and_keeps_baseline(workspace):
    make_client, directory, _, vid = workspace
    client = make_client(); s = client.get(f'/api/sessions/{vid}').json()
    original = (directory / 'frames.parquet').read_bytes()
    edits = s['review']['edits']
    edits['labels'] = [{'start_s': 0, 'end_s': 1, 'state': 'Listing/LORR', 'reviewer': 'R'}]
    review = client.put(f'/api/sessions/{vid}/review', json={
        'revision': 0, 'baseline': s['baseline'], 'reviewer': 'R', 'edits': edits, 'decision': None}).json()
    r = client.post(f'/api/sessions/{vid}/rerun', json={'revision': review['revision'], 'baseline': s['baseline'], 'reviewer': 'R'})
    assert r.status_code == 200
    assert not r.json()['stale']['measurements']
    assert r.json()['stale']['predictions']
    assert (directory / 'frames.parquet').read_bytes() == original


def test_parallel_saves_have_one_winner_and_source_replacement_is_detected(workspace):
    make_client, directory, sources, vid = workspace
    client = make_client(); s = client.get(f'/api/sessions/{vid}').json()
    body = {'revision': 0, 'baseline': s['baseline'], 'reviewer': 'R', 'edits': s['review']['edits'], 'decision': 'ACCEPTED'}
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: client.put(f'/api/sessions/{vid}/review', json=body).status_code, range(2)))
    assert sorted(responses) == [200, 409]
    reloaded = make_client().get(f'/api/sessions/{vid}').json()
    assert reloaded['review']['decision'] == 'ACCEPTED'
    source = sources / f'{vid}.mp4'
    source.write_bytes(source.read_bytes() + b'changed')
    assert client.get(f'/api/sessions/{vid}').status_code == 409


def test_failed_atomic_replace_preserves_previous_review(workspace, monkeypatch):
    from prepds.webapp import workspace as store
    make_client, directory, _, vid = workspace
    client = make_client(); s = client.get(f'/api/sessions/{vid}').json()
    body = {'revision': 0, 'baseline': s['baseline'], 'reviewer': 'R', 'edits': s['review']['edits'], 'decision': None}
    first = client.put(f'/api/sessions/{vid}/review', json=body).json()
    previous = (directory / 'review.json').read_bytes()
    body['revision'] = first['revision']
    monkeypatch.setattr(store.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError):
        client.put(f'/api/sessions/{vid}/review', json=body)
    assert (directory / 'review.json').read_bytes() == previous
    assert not (directory / 'review.json.tmp').exists()


def test_actual_processing_uses_catalog_and_preserves_existing_work(workspace):
    make_client, directory, sources, vid = workspace
    from prepds import review_store
    from prepds.models import Trial, MatchStatus
    manifest = review_store.load_manifest(directory)
    trial = Trial(manifest.subject_id, manifest.sex, 'Casper', None, manifest.compound,
                  manifest.concentration_mM, dt.date(2026, 3, 1), 20.0,
                  sources / f'{vid}.mp4', MatchStatus.MATCHED)
    pd.DataFrame([trial.to_dict()]).to_parquet(directory.parent.parent / 'trials_catalog.parquet')
    env = sources.parent / 'test.env'; env.write_text('')
    settings = load_settings(env_file=env, environ={})
    client = TestClient(create_app(directory.parent, sources.parent / 'accepted', video_dir=sources,
                                   settings=settings, allowed_hosts=('testserver',)))
    assert client.post(f'/api/sessions/{vid}/process', json={'reviewer': 'R'}).status_code == 409
    for name in ('manifest.json', 'frames.parquet', 'segments.csv', 'strip.png'):
        (directory / name).unlink()
    r = client.post(f'/api/sessions/{vid}/process', json={'reviewer': 'R'})
    assert r.status_code == 202
    assert client.get(f'/api/sessions/{vid}/processing').json()['status'] == 'processed'
    assert client.get(f'/api/sessions/{vid}').json()['has_tracking']
    assert (directory / 'frames.parquet').is_file()


def test_configured_chat_round_trip_and_invalid_service_response(workspace):
    make_client, directory, sources, _ = workspace
    class Handler(BaseHTTPRequestHandler):
        bad = False
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/api/ask'
            self.send_response(200); self.end_headers()
            payload = {} if self.bad else {'answer': 'from service', 'tool_calls': [],
                       'history': body['history'] + [{'role': 'assistant', 'content': 'from service'}]}
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        client = TestClient(create_app(directory.parent, sources.parent / 'accepted', video_dir=sources,
                                      chat_url=f'http://127.0.0.1:{server.server_port}', allowed_hosts=('testserver',)))
        result = client.post('/api/ask', json={'question': 'hello', 'history': []})
        assert result.status_code == 200 and result.json()['answer'] == 'from service'
        Handler.bad = True
        assert client.post('/api/ask', json={'question': 'q'}).status_code == 503
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_model_inference_contract_and_stale_revision(workspace, tmp_path, monkeypatch):
    from prepds.webapp import workspace as store
    from types import SimpleNamespace
    make_client, directory, sources, vid = workspace
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DCS_DB_PATH', 'workbook.xlsx')
    monkeypatch.setenv('DCS_CONFIG', 'training.yaml')
    root = tmp_path / 'dcs'; (root / 'src/dcs').mkdir(parents=True)
    run = tmp_path / 'run'; (run / 'model').mkdir(parents=True)
    (run / 'model/model_info.json').write_text('{}')
    def subprocess(command, **kwargs):
        assert kwargs['env']['DCS_DB_PATH'] == str(tmp_path / 'workbook.xlsx')
        assert kwargs['env']['DCS_CONFIG'] == str(tmp_path / 'training.yaml')
        output = Path(command[command.index('--out') + 1])
        if 'featurize' in command:
            snapshot = Path(command[command.index('--videos') + 1]) / vid
            frames = pd.read_parquet(snapshot / 'frames.parquet')
            assert frames['state'].dtype.name == 'category'
            pd.DataFrame({'video_id': [vid]}).to_parquet(output)
        else:
            pd.DataFrame({'video_id': [vid], 'predicted': ['test_class'], 'p:test_class': [1.0]}).to_csv(output, index=False)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(store.subprocess, 'run', subprocess)
    client = TestClient(create_app(directory.parent, sources.parent / 'accepted', video_dir=sources,
                                   classifier_root=root, model_run=run, allowed_hosts=('testserver',)))
    s = client.get(f'/api/sessions/{vid}').json()
    body = {'revision': s['review']['revision'], 'baseline': s['baseline'], 'reviewer': 'R'}
    r = client.post(f'/api/sessions/{vid}/predict', json=body)
    assert r.status_code == 200
    assert r.json()['predictions']['predicted'] == 'test_class'
    assert r.json()['model_provenance']['run'] == 'run'
    assert client.post(f'/api/sessions/{vid}/predict', json=body).status_code == 409
    updated = r.json()
    edits = updated['review']['edits']
    edits['finalResult'] = {'compound': 'manual_class', 'dose': '1 mM', 'reviewer': 'R'}
    saved = client.put(f'/api/sessions/{vid}/review', json={
        **body, 'revision': updated['review']['revision'], 'edits': edits, 'decision': None})
    assert saved.status_code == 200
    assert client.get(f'/api/sessions/{vid}').json()['predictions']['predicted'] == 'test_class'


def test_built_frontend_and_prediction_validation(workspace, tmp_path):
    make_client, directory, sources, vid = workspace
    frontend = tmp_path / 'dist'; (frontend / 'assets').mkdir(parents=True)
    (frontend / 'index.html').write_text('<html>React app</html>')
    (frontend / 'assets/app.js').write_text('console.log(1)')
    predictions = tmp_path / 'predictions.csv'
    pd.DataFrame({'video_id': [vid], 'predicted': ['bad'], 'p:bad': [2.0]}).to_csv(predictions, index=False)
    client = TestClient(create_app(directory.parent, sources.parent / 'accepted', video_dir=sources,
                                   frontend_dir=frontend, predictions_path=predictions, allowed_hosts=('testserver',)))
    assert 'React app' in client.get('/').text
    assert client.get('/assets/app.js').status_code == 200
    s = client.get(f'/api/sessions/{vid}').json()
    assert s['predictions'] is None and s['warnings']


@pytest.mark.parametrize('url', ['https://remote.example', 'http://remote.example', 'http://user:pass@localhost:8010', 'http://localhost:8010/?token=secret'])
def test_chat_configuration_cannot_send_data_to_remote_services(tmp_path, url):
    from prepds.config import ConfigError
    with pytest.raises(ConfigError, match='loopback'):
        create_app(tmp_path / 'processed', tmp_path / 'accepted', chat_url=url)
