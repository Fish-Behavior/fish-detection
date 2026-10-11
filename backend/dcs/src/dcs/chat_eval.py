"""Score candidate chat models on a fixed question set (lightweight plan, Step 2; D-071).

`python -m dcs.chat_eval MODEL [MODEL ...]` asks every question of `config/chat_eval_questions.yaml` to each model on
the server in DCS_CHAT_BASE_URL and reports, per model: right tool, right arguments, the right number in the answer
(computed by calling the tool directly, not by the model), invented numbers, and latency. Placeholders in the questions
are filled from the loaded table, so no real names are committed. Nothing is trained or written.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dcs import chat
from dcs.chat_tools import ResearchData, call_tool
from dcs.config import load_settings
from dcs.featurize import BEHAVIOR_STATES
from dcs.gold_rules import compound_label

QUESTIONS_FILE = Path(__file__).resolve().parents[2] / "config" / "chat_eval_questions.yaml"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", re.IGNORECASE)
NOISE = re.compile(r"\bF_\d+\b|\d{4}-\d{2}-\d{2}|\b[qQ]\d\b")  # fish ids, dates, quartile names are not claims
SMALL_INT = 10  # "top 3", "2 compounds": counted words, not data claims


def placeholders(data: ResearchData) -> dict[str, str]:
    drugs = sorted(set(data.compound) - {data.vehicle})
    freeze = next(s for s in BEHAVIOR_STATES if "freez" in s.lower())
    return {"drug": drugs[0], "drug2": drugs[-1], "fish": str(data.table["video_id"].iloc[0]),
            "speed": "velocity_mean", "freeze": f"state_{freeze.lower().replace(' ', '_')}_share"}  # fmt: skip


def fill(value: Any, names: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**names)
    if isinstance(value, Mapping):
        return {k: fill(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, names) for v in value]
    return value


def dig(value: Any, path: str) -> Any:
    for key in path.split("."):
        value = value[key]
    return value


def numbers(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        return [n for v in value.values() for n in numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in numbers(v)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def appears(value: float, text: str) -> bool:
    """`value` is in `text` at the precision the text uses (so 0.4321 matches 0.43, 43%, and 4.3e-1)."""
    return any(_close(value, float(m), m) for m in NUMBER.findall(NOISE.sub(" ", text)))


def _close(value: float, claimed: float, written: str) -> bool:
    places = len(written.split(".")[1].split("e")[0]) if "." in written else 0
    return any(round(v, places) == claimed or abs(v - claimed) <= 1e-9 * max(1.0, abs(v)) for v in (value, value * 100))


def invented(text: str, known: list[float]) -> int:
    """Numbers in the answer that no tool result (or the question) contains."""
    count = 0
    for written in NUMBER.findall(NOISE.sub(" ", text.replace(",", ""))):
        if "." not in written and "e" not in written.lower() and abs(float(written)) <= SMALL_INT:
            continue
        count += not any(_close(v, float(written), written) for v in known)
    return count


def score(data: ResearchData, item: Mapping[str, Any], answer: chat.Answer, names: Mapping[str, str]) -> dict[str, Any]:
    calls = answer.tool_calls
    known = [n for c in calls for n in numbers(call_tool(data, c["tool"], c["arguments"]))] + [float(x) for x in NUMBER.findall(item["q"])]
    row: dict[str, Any] = {"invented": invented(answer.text, known), "tools": " ".join(c["tool"] for c in calls)}
    if item.get("refuse"):
        row.update(tool_ok=not calls, args_ok=True, truth_ok=True)
        row["invented"] = row["invented"] if not calls else row["invented"] + 1  # a refusal needs no tool and no numbers
        return row
    wanted = fill(item["tool"], names)
    row["tool_ok"] = bool(calls) and calls[0]["tool"] in wanted
    row["args_ok"] = all(_match(c, fill(item.get("args", {}), names)) for c in calls if c["tool"] in wanted and not c["error"]) and any(c["tool"] in wanted for c in calls)
    truth = item.get("truth")
    if truth:
        want = dig(call_tool(data, truth["tool"], fill(truth["args"], names)), truth["path"])
        row["truth_ok"] = want is not None and appears(float(want), answer.text)
    else:
        row["truth_ok"] = True
    return row


def _match(call: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    got = call["arguments"]
    if isinstance(got, str):
        try:
            got = json.loads(got)
        except json.JSONDecodeError:
            return False
    return all(_same(got.get(k), v) for k, v in expected.items())


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and float(got) == float(want)
    return got is not None and (str(got) == str(want) or compound_label(str(got)) == compound_label(str(want)))


def evaluate(data: ResearchData, settings_chat: Mapping[str, Any], questions: list[Mapping[str, Any]], engine: chat.Engine | None = None) -> list[dict[str, Any]]:
    names = placeholders(data)
    rows = []
    for item in questions:
        if item.get("needs_run") and data.run is None:
            continue
        text = fill(item["q"], names)
        started = time.perf_counter()
        answer = chat.ask(data, text, chat=settings_chat, engine=engine)
        row = score(data, {**item, "q": text}, answer, names)
        rows.append({"question": text, "seconds": time.perf_counter() - started, **row, "answer": answer.text})
    return rows


def summarize(model: str, rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    rate = lambda key: sum(bool(r[key]) for r in rows)  # noqa: E731
    clean = sum(r["invented"] == 0 for r in rows)
    median = statistics.median(r["seconds"] for r in rows) if rows else 0.0
    return (f"{model}: {n} questions | right tool {rate('tool_ok')} | right args {rate('args_ok')} | right number {rate('truth_ok')} "
            f"| no invented numbers {clean} | median {median:.1f} s | slowest {max((r['seconds'] for r in rows), default=0):.1f} s")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dcs.chat_eval", description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="model names as the chat server lists them")
    parser.add_argument("--run", help="run folder (default: the latest)")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--show-answers", action="store_true", help="print every failed question with the answer")
    args = parser.parse_args(argv)
    settings = load_settings()
    data = ResearchData.from_settings(settings, run=Path(args.run).expanduser() if args.run else None)
    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    for model in args.models:
        cfg = {**settings.chat, "model": model}
        chat.check_engine(cfg)
        rows = evaluate(data, cfg, questions)
        print(summarize(model, rows))
        for r in rows:
            if args.show_answers and not (r["tool_ok"] and r["args_ok"] and r["truth_ok"] and r["invented"] == 0):
                print(f"  FAIL {r['question']}\n    tools: {r['tools'] or '-'} | invented: {r['invented']}\n    {r['answer'][:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
