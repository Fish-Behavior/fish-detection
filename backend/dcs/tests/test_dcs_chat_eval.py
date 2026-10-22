"""Chat-model scorer (lightweight plan Step 2): a scripted engine that answers correctly passes, an inventing one fails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dcs.chat_eval import QUESTIONS_FILE, evaluate, invented, numbers, placeholders
from dcs.chat_tools import ResearchData, call_tool
from dcs.config import load_settings
from dcs.trainset import load_table

CHAT = {**load_settings(environ={}).params["chat"], "base_url": "http://127.0.0.1:1/v1", "model": "x"}
QUESTIONS = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]


@pytest.fixture(scope="module")
def data(tiny_table: Path) -> ResearchData:
    table, described = load_table(tiny_table)
    return ResearchData(table, described, dict(load_settings().training), None, None)


def test_questions_are_well_formed(data: ResearchData) -> None:
    names = placeholders(data)
    for item in QUESTIONS:
        item["q"].format(**names)
        assert item.get("refuse") or item["tool"]


def test_scorer_passes_a_good_model_and_fails_a_bad_one(data: ResearchData) -> None:
    names = placeholders(data)
    args = {"compound": names["drug"], "feature": names["speed"]}
    g = call_tool(data, "compare_to_vehicle", args)["hedges_g"]
    question = [q for q in QUESTIONS if q.get("truth", {}).get("path") == "hedges_g" and q["truth"]["args"]["feature"] == "{speed}"][:1]
    calls = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "compare_to_vehicle", "arguments": args}}]}

    def engine(answer: str) -> Any:
        replies = iter([calls, {"role": "assistant", "content": answer}])
        return lambda *_: next(replies)

    good = evaluate(data, CHAT, question, engine(f"Hedges g is {g:.2f}."))[0]
    assert good["tool_ok"] and good["args_ok"] and good["truth_ok"] and good["invented"] == 0
    bad = evaluate(data, CHAT, question, engine("Hedges g is 9.87 with p 0.0001."))[0]
    assert not bad["truth_ok"] and bad["invented"] == 2

    refuse = [{"q": "Write a poem.", "refuse": True}]
    poem = lambda *_: {"role": "assistant", "content": "Fish swim in 3 rows."}  # noqa: E731
    assert evaluate(data, CHAT, refuse, poem)[0]["tool_ok"]


def test_a_lookup_before_the_comparison_is_not_a_wrong_tool(data: ResearchData) -> None:
    names = placeholders(data)
    args = {"compound": names["drug"], "feature": names["speed"]}
    g = call_tool(data, "compare_to_vehicle", args)["hedges_g"]
    question = [q for q in QUESTIONS if q.get("truth", {}).get("path") == "hedges_g" and q["truth"]["args"]["feature"] == "{speed}"][:1]
    lookup = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "find_features", "arguments": {"search": "speed"}}}]}
    compare = {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "compare_to_vehicle", "arguments": args}}]}
    replies = iter([lookup, compare, {"role": "assistant", "content": f"Hedges g is {g:.2f}."}])
    row = evaluate(data, CHAT, question, lambda *_: next(replies))[0]
    assert row["tool_ok"] and row["args_ok"] and row["truth_ok"]


def test_numbers_inside_tool_text_and_compound_codes_are_not_invented() -> None:
    assert numbers({"scheme_A": "0.469 ± 0.022", "doses": ["0.03+0.01"]}) == [0.469, 0.022, 0.03, 0.01]
    assert invented("FD-2-67 and MTA-5-62 differ: 0.469 vs 0.022.", [0.469, 0.022]) == 0
    assert invented("FD-2-67 scored 0.512.", [0.469]) == 1
