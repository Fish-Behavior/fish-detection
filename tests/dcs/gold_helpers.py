"""Helpers for the gold-reader tests (test_dcs_gold*.py); not a test module itself.

The synthetic generator writes the accepted gold layout only. `to_processed` turns such a set
into the unreviewed prepds output layout (`<out>/processed/<video_id>/` plus
`<out>/trials_catalog.parquet`), so both reader sources are tested on the same fish.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from dcs import schema
from dcs.gold import GoldSet
from dcs.synthetic import SynthResult

TableChange = Callable[[pd.DataFrame], pd.DataFrame]
DAMAGE_BYTES = 24  # overwrites the column's first page header, so its data can no longer be decoded


def only(result: SynthResult, knob: str) -> str:
    """The one fish a single-fish knob was applied to."""
    (video_id,) = result.targets[knob]
    return video_id


def dropped_reasons(gold: GoldSet) -> dict[str, str]:
    """video_id -> drop reason."""
    return dict(zip(gold.dropped["video_id"], gold.dropped["reason"]))


def damage_column(path: Path, column: str) -> None:
    """Damage one column's pages of a parquet file in place; the footer (schema) stays readable."""
    metadata = pq.ParquetFile(path).metadata
    chunk = metadata.row_group(0).column(metadata.schema.to_arrow_schema().names.index(column))
    offset = chunk.dictionary_page_offset or chunk.data_page_offset
    data = bytearray(path.read_bytes())
    data[offset : offset + DAMAGE_BYTES] = b"\xff" * DAMAGE_BYTES
    path.write_bytes(bytes(data))


def to_processed(result: SynthResult, out: Path, status: str = schema.REVIEW_PROCESSED_AUTO) -> Path:
    """Copy every fish into `<out>/processed/<video_id>/` with manifest status `status`; write the catalog."""
    index = pd.read_parquet(result.index_path)
    processed = out / schema.PROCESSED_DIR_NAME
    processed.mkdir(parents=True)
    for video_id in index["video_id"]:
        shutil.copytree(result.accepted_dir / video_id, processed / video_id)
        (processed / video_id / schema.PROVENANCE_FILE).unlink(missing_ok=True)  # prepds writes it only on Accept
        set_manifest(processed / video_id, review_status=status)
    catalog = index.assign(match_status=schema.MATCH_MATCHED)[list(schema.CATALOG_COLUMNS)]
    catalog.to_parquet(out / schema.CATALOG_FILE, index=False)
    return out


def set_manifest(folder: Path, **changes: Any) -> None:
    """Change keys of `<folder>/manifest.json` in place."""
    path = folder / schema.MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def rewrite_table(path: Path, change: TableChange) -> None:
    """Read a parquet table, apply `change` and write it back."""
    change(pd.read_parquet(path)).to_parquet(path, index=False)


def rewrite_segments(folder: Path, change: TableChange) -> None:
    """Read `<folder>/segments.csv`, apply `change` and write it back."""
    path = folder / schema.SEGMENTS_FILE
    change(pd.read_csv(path)).to_csv(path, index=False)
