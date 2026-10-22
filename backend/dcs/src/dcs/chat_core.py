"""Tool registry and research data for the chat (plan U19; D-071).

`ResearchData` is what every tool reads; `@tool` registers a function in `TOOLS`; `call_tool` runs one and turns a bad
argument or a tool bug into `{"error": ...}` for the model to read. The tools themselves live in `dcs.chat_tools`
(the data) and `dcs.chat_results` (training-run results); import `dcs.chat_tools` to get all of them registered.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcs import schema
from dcs.config import Settings
from dcs.config_rules import SOURCE_PROCESSED
from dcs.gold_rules import compound_label
from dcs.trainset import dose_label, load_table


class ToolError(ValueError):
    """A bad argument; the message says what is allowed."""


@dataclass(frozen=True)
class ResearchData:
    table: pd.DataFrame  # the training table (one row per fish)
    described: Mapping[str, Any]  # its schema JSON
    training: Mapping[str, Any]  # training settings (vehicle name)
    gold_dir: Path | None  # holds <video_id>/frames.parquet and segments.csv
    run: Path | None  # a `dcs train` run folder

    @classmethod
    def from_settings(cls, settings: Settings, run: Path | None = None) -> ResearchData:
        """The table from DCS_TABLE, the video folders of the gold source, and `run` or the latest run folder."""
        table, described = load_table(settings.require("table"))
        if described["gold_source"] == SOURCE_PROCESSED:
            processed = settings.paths.processed_dir
            gold_dir = None if processed is None else processed / schema.PROCESSED_DIR_NAME
        else:
            gold_dir = settings.paths.accepted_dir
        if run is None:
            runs = sorted((settings.paths.output_dir / "training").glob("*/run_info.json"))
            run = runs[-1].parent if runs else None
        return cls(table, described, dict(settings.training), gold_dir, run)

    @property
    def compound(self) -> pd.Series:
        return self.table["compound"].map(compound_label)

    @property
    def dose(self) -> pd.Series:
        return self.table["concentration_mM"].map(dose_label)

    @property
    def vehicle(self) -> str:
        return compound_label(self.training["vehicle_compound"])


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    run: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {}


def tool(description: str, required: tuple[str, ...] = (), **properties: Any) -> Callable[..., Any]:
    def register(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        TOOLS[function.__name__] = Tool(function.__name__, description, properties, required, function)
        return function

    return register


def tool_schemas() -> list[dict[str, Any]]:
    """The tools in the OpenAI function-calling format (also understood by Ollama, llama.cpp and vLLM)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {"type": "object", "properties": t.properties, "required": list(t.required)},
            },
        }
        for t in TOOLS.values()
    ]


def call_tool(data: ResearchData, name: str, arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Run one tool; bad names, arguments or JSON come back as `{"error": ...}` for the model to fix."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; tools: {', '.join(TOOLS)}"}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        return plain(TOOLS[name].run(data, **args))
    except (ToolError, json.JSONDecodeError, TypeError) as error:
        return {"error": str(error)}
    except Exception as error:  # noqa: BLE001 (a tool bug must not end `dcs ask` or turn into a 500)
        return {"error": f"{name} failed: {type(error).__name__}: {error}"}


def num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def plain(value: Any) -> Any:
    """JSON-safe: numpy scalars to Python, tuples to lists, NaN to None, dates to text."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
