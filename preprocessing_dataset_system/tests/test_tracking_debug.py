"""Debug-overlay video export for visual tracking spot-checks (T035)."""

from __future__ import annotations

from pathlib import Path

from prepds.tracking import render_debug_overlay, track_video
from prepds.video_io import probe

FIXTURES = Path(__file__).parent / "fixtures"
SYNTH_VIDEO = FIXTURES / "synth_tiny.mp4"


def test_render_debug_overlay_writes_playable_video_same_length(tmp_path: Path) -> None:
    tracks = track_video(SYNTH_VIDEO)
    output_path = tmp_path / "debug.mp4"

    render_debug_overlay(SYNTH_VIDEO, output_path, tracks)

    assert output_path.is_file()
    asset = probe(output_path)
    assert asset.frame_count == len(tracks)
    assert asset.resolution == (304, 240)
