"""Scene review: a browser page to check and correct each video's waterline and ROI.

`python -m fishbehavior scene-review` brings the scene results up to date, then opens
a page (served only on this computer, 127.0.0.1) showing one video at a time:
its empty-beaker background, where the fish moved (red tint), the waterline (blue)
and the ROI (green). A person clicks to move the waterline, drags to redraw the ROI,
or confirms the automatic result. The cursor position is shown in ORIGINAL video
pixels, so no zoom arithmetic is needed. Every change is saved straight into
``<FISH_OUTPUT_DIR>/scene/overrides.yaml`` and applied when the page is closed.

With ``--no-serve`` (e.g. on Google Colab) the page is only written to
``scene/review.html``; its "Download overrides.yaml" button gives the file to put in
the scene folder.

Nothing leaves the computer: the page embeds the images and has no external links.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

from fishbehavior.catalog import STATUS_MATCHED
from fishbehavior.config import ConfigError
from fishbehavior.roi import (
    FLAG_LOW_CONFIDENCE,
    FLAG_NO_ACTIVITY,
    OVERRIDES_FILE,
    json_path,
    load_overrides,
    validate_overrides,
)

log = logging.getLogger(__name__)

REVIEW_FILE = "review.html"
TEMPLATE = "review_page.html"  # packaged next to this module
DATA_PLACEHOLDER = "__REVIEW_DATA__"

# Hints shown on the page and used to sort videos that most likely need a look first.
HINT_TOP_FRACTION = 0.10  # waterline or ROI top in the top 10% of the frame is suspicious


# ---------------------------------------------------------------------------
# 1. Page data
# ---------------------------------------------------------------------------


def _image_data_url(path: Path, jpeg: bool) -> str | None:
    """Embed an image in the page (JPEG for the photo-like background keeps the page small)."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    ext, mime, options = (".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, 92]) if jpeg else (".png", "image/png", [])
    ok, buffer = cv2.imencode(ext, image, options)
    return f"data:{mime};base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}" if ok else None


def review_hints(record: dict[str, Any]) -> list[str]:
    """Plain-language reasons why a person should look at this video's automatic result."""
    height = record["video"]["height"]
    auto = record["auto"]
    hints = []
    if FLAG_LOW_CONFIDENCE in record.get("flags", []):
        hints.append("waterline edge is weak")
    if FLAG_NO_ACTIVITY in record.get("flags", []):
        hints.append("fish hardly moved")
    if auto["waterline_y"] < HINT_TOP_FRACTION * height:
        hints.append("waterline at the very top of the frame")
    if auto["roi"][1] == 0:
        hints.append("ROI reaches the top edge")
    return hints


def collect_items(scene_dir: Path, trials: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    """One page item per matched video that has a scene result; also the files without one."""
    items, missing = [], []
    for row in trials[trials["video_status"] == STATUS_MATCHED].itertuples():
        for path in (Path(p) for p in row.video_paths.split(";") if p):
            result = json_path(scene_dir, path)
            if not result.is_file():
                missing.append(path.name)
                continue
            record = json.loads(result.read_text(encoding="utf-8"))
            items.append({
                "file": record["file"],
                "subject_id": record["subject_id"],
                "width": record["video"]["width"],
                "height": record["video"]["height"],
                "confidence": record["waterline_confidence"],
                "auto": record["auto"],
                "base_roi": record["base_roi"],
                "hints": review_hints(record),
                "background": _image_data_url(scene_dir / record["background_image"], jpeg=True),
                "activity": _image_data_url(scene_dir / record["activity_image"], jpeg=False),
            })
    # Videos with hints first (they most likely need fixing), then by file name.
    items.sort(key=lambda item: (not item["hints"], item["file"]))
    return items, missing


def render_page(items: list[dict[str, Any]], overrides: dict[str, dict[str, Any]],
                params: dict[str, Any], serve: bool) -> str:
    """The review page with its data embedded (one self-contained HTML file)."""
    files = {item["file"] for item in items}
    data = {
        "items": items,
        # Current entries for the videos on the page; the others are kept untouched on save.
        "overrides": {name: entry for name, entry in overrides.items() if name in files},
        "other_overrides": {name: entry for name, entry in overrides.items() if name not in files},
        "margin_fraction": float(params["waterline_margin_fraction"]),
        "serve": serve,
    }
    template = resources.files("fishbehavior").joinpath(TEMPLATE).read_text(encoding="utf-8")
    # "</" inside a <script> block would end it early; JSON allows the escaped form.
    return template.replace(DATA_PLACEHOLDER, json.dumps(data).replace("</", "<\\/"))


# ---------------------------------------------------------------------------
# 2. Saving overrides from the page
# ---------------------------------------------------------------------------


def check_in_frame(entries: dict[str, dict[str, Any]], sizes: dict[str, tuple[int, int]]) -> None:
    """Refuse values outside the video frame (the page prevents this; this guards the file)."""
    for name, entry in entries.items():
        width, height = sizes[name]
        if "waterline_y" in entry and not 0 <= entry["waterline_y"] < height:
            raise ConfigError(f"{name}: waterline_y {entry['waterline_y']} is outside the frame (height {height})")
        if "roi" in entry:
            x0, y0, x1, y1 = entry["roi"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ConfigError(f"{name}: roi {entry['roi']} is outside the {width}x{height} frame")


def save_page_overrides(scene_dir: Path, page_entries: Any, sizes: dict[str, tuple[int, int]]) -> int:
    """Replace the entries of the page's videos in overrides.yaml, keeping all other entries.

    `page_entries` is the page's full state for its videos: a video missing from it has
    no override (reset to automatic). Returns the number of entries written.
    """
    entries = validate_overrides(page_entries, "review page")
    unknown = set(entries) - set(sizes)
    if unknown:
        raise ConfigError(f"review page: videos not on the page: {', '.join(sorted(unknown))}")
    check_in_frame(entries, sizes)
    merged = {name: entry for name, entry in load_overrides(scene_dir).items() if name not in sizes}
    merged.update(entries)
    write_overrides(scene_dir, merged)
    return len(merged)


def write_overrides(scene_dir: Path, overrides: dict[str, dict[str, Any]]) -> None:
    """Write overrides.yaml atomically (a crash mid-write never leaves half a file)."""
    header = (
        "# Manual scene corrections, per video file name. Written by `scene-review`; can be edited by hand.\n"
        "#   waterline_y: row of the water surface (original video pixels)\n"
        "#   roi: [x0, y0, x1, y1] region where the fish can be (x1/y1 exclusive)\n"
        "#   checked: true = a person looked at this video and accepts its result\n"
    )
    body = yaml.safe_dump(dict(sorted(overrides.items())), sort_keys=False, default_flow_style=None)
    path = scene_dir / OVERRIDES_FILE
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(header + (body if overrides else ""), encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# 3. Local server
# ---------------------------------------------------------------------------


def make_server(scene_dir: Path, page: str, sizes: dict[str, tuple[int, int]], port: int) -> ThreadingHTTPServer:
    """An HTTP server on 127.0.0.1 that serves the page and saves overrides it posts.

    Routes: GET / (the page), POST /save (JSON {file: entry}), POST /done (stop serving).
    """
    lock = threading.Lock()  # one save at a time

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict[str, Any]) -> None:
            self._reply(status, json.dumps(data).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path in ("/", "/index.html"):
                self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._reply(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/done":
                self._json(200, {"ok": True})
                # shutdown() waits for serve_forever to return, so call it from another thread.
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/save":
                self._json(404, {"ok": False, "error": "unknown address"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                entries = json.loads(self.rfile.read(length) or b"{}")
                with lock:
                    count = save_page_overrides(scene_dir, entries, sizes)
            except (ConfigError, ValueError) as error:  # ValueError includes bad JSON
                self._json(400, {"ok": False, "error": str(error)})
                return
            self._json(200, {"ok": True, "entries": count})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            log.debug("review server: " + format, *args)  # keep the terminal quiet

    # 127.0.0.1 only: the page is never reachable from other computers.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_review(server: ThreadingHTTPServer, open_browser: bool = True) -> None:
    """Serve until the page's "Finish" button is pressed or Ctrl+C."""
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Review page: {url}   (press Finish on the page, or Ctrl+C here, when done)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
