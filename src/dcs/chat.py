"""Research chat: questions in English, answers built from the query tools (plan U20; D-071, D-073).

A local open-weights model (any OpenAI-compatible server with tool calling: Ollama, llama.cpp, vLLM) reads the
question, calls `chat_tools` for every number, and writes the answer. Nothing is trained. The client is the standard
library only. A server address that is not this machine is refused unless `chat.allow_remote` is true, so data
summaries cannot leave the box by accident. `ask` serves both the terminal (`dcs ask`) and the API (`chat_server`).
"""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dcs.chat_tools import ResearchData, call_tool, tool_schemas
from dcs.config import ConfigError
from dcs.config_rules import SOURCE_PROCESSED
from dcs.featurize import BEHAVIOR_STATES

API_KEY_VARIABLE = "DCS_CHAT_API_KEY"
LOCAL_NAMES = ("localhost",)
FINAL_NUDGE = "Answer the question now from the tool results above; do not call more tools."
SYSTEM_PROMPT = f"""You are the research assistant of a zebrafish drug-exposure behavior study. Researchers ask you, in \
English, about how fish behaved under each compound and dose, about single fish, and about the classifier's results.

Rules:
- Get every number from the tools. Never invent numbers, fish ids, compounds, doses or dates. If the tools cannot \
answer, say so and say what would be needed.
- Before comparing behavior, look up the measure's name with find_features (freezing, speed, anxiety, turning ...).
- When you compare groups, give n, the means or medians, the effect size (Hedges' g) and p (or q for many measures), \
and say plainly how strong the evidence is: small groups and many tests make chance findings likely.
- Repeat the caveats the tools return. The data may be UNREVIEWED (temporarily accepted pipeline output), and \
compounds were recorded on different dates and cameras; pixel-based measures depend on the camera.
- For model results, scheme A (whole recording dates held out) is the honest score; scheme B can be won by \
recognizing the date; date_only and the within-date permuted score are the baselines. Explain a verdict with its \
reason.
- Behavior states: {", ".join(BEHAVIOR_STATES)}. "Undetermined" is unknown time, not a behavior.
- If a tool returns an error, correct the arguments using its message, or tell the researcher what is missing.
- Answer in plain English for biologists: short paragraphs or small tables, the numbers first, then what they mean.
"""

Engine = Callable[[Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


class EngineError(ConfigError):
    """The chat server cannot be reached or answered with an error; the message says what to check."""


@dataclass(frozen=True)
class Answer:
    text: str
    tool_calls: list[dict[str, Any]]  # tool, arguments, error (None when it worked)
    history: list[dict[str, str]]  # user and assistant turns, to send back with the next question


def check_engine(chat: Mapping[str, Any]) -> None:
    """A model name is set, and the server is this machine unless `allow_remote` says otherwise."""
    if not chat["model"]:
        raise ConfigError(
            "chat.model is not set: put the model's name (as your chat server lists it, e.g. `ollama list`) in "
            "your override file under chat: model:"
        )
    host = urlparse(chat["base_url"]).hostname or ""
    if not _is_local(host) and not chat["allow_remote"]:
        raise ConfigError(
            f"chat.base_url {chat['base_url']} is not this machine, so questions and data summaries would leave "
            "it. Set chat.allow_remote: true only if your data agreement allows it."
        )


def complete(chat: Mapping[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """One request to `<base_url>/chat/completions`; returns the reply message (content and/or tool_calls)."""
    body: dict[str, Any] = {"model": chat["model"], "messages": messages, "temperature": chat["temperature"], "stream": False}
    if tools:
        body["tools"] = tools
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(API_KEY_VARIABLE)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = chat["base_url"].rstrip("/") + "/chat/completions"
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=chat["timeout_s"]) as response:  # noqa: S310 (http(s) only, by rule)
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise EngineError(f"The chat server at {url} answered {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise EngineError(
            f"Cannot reach the chat server at {url} ({getattr(error, 'reason', error)}). Start it (for Ollama: "
            "`ollama serve`, then `ollama pull <model>`), or fix chat.base_url."
        ) from None
    try:
        return payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise EngineError(f"The chat server at {url} sent a reply without a message: {str(payload)[:300]}") from None


def ask(
    data: ResearchData,
    question: str,
    history: Sequence[Mapping[str, str]] = (),
    *,
    chat: Mapping[str, Any],
    engine: Engine | None = None,
) -> Answer:
    """Answer one question: the model calls tools until it can answer, at most `chat.max_tool_rounds` times."""
    engine = engine or complete  # looked up at call time, so tests can replace `chat.complete`
    turns = [{"role": t["role"], "content": t["content"]} for t in history]
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT + _context(data)}, *turns, {"role": "user", "content": question}]
    calls: list[dict[str, Any]] = []
    tools = tool_schemas()
    for _ in range(chat["max_tool_rounds"]):
        reply = engine(chat, messages, tools)
        requested = reply.get("tool_calls") or []
        if not requested:
            return _answer(reply, question, turns, calls)
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": requested})
        for number, call in enumerate(requested):
            function = call.get("function", {})
            name, arguments = function.get("name", ""), function.get("arguments") or {}
            result = call_tool(data, name, arguments)
            calls.append({"tool": name, "arguments": arguments, "error": result.get("error")})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{number}", "name": name, "content": json.dumps(result)})
    reply = engine(chat, [*messages, {"role": "user", "content": FINAL_NUDGE}], [])
    return _answer(reply, question, turns, calls)


def session(data: ResearchData, chat: Mapping[str, Any], show_tools: bool, read: Callable[[str], str] = input) -> None:
    """An interactive terminal session; an empty line, `exit` or Ctrl-D ends it."""
    print("Ask about compounds, single fish (e.g. F_0042) or the model results. Empty line or `exit` to stop.")
    history: list[dict[str, str]] = []
    while True:
        try:
            question = read("\n> ").strip()
        except EOFError:
            return
        if question.lower() in ("", "exit", "quit"):
            return
        answer = ask(data, question, history, chat=chat)
        print_answer(answer, show_tools)
        history = answer.history


def print_answer(answer: Answer, show_tools: bool) -> None:
    if show_tools:
        for call in answer.tool_calls:
            status = f"error: {call['error']}" if call["error"] else "ok"
            print(f"  [tool] {call['tool']}({json.dumps(call['arguments']) if not isinstance(call['arguments'], str) else call['arguments']}) -> {status}")
    print(answer.text)


def _answer(reply: Mapping[str, Any], question: str, turns: list[dict[str, str]], calls: list[dict[str, Any]]) -> Answer:
    text = (reply.get("content") or "").strip() or "(the model gave no answer)"
    return Answer(text, calls, [*turns, {"role": "user", "content": question}, {"role": "assistant", "content": text}])


def _context(data: ResearchData) -> str:
    """A few facts about the loaded data, so the model knows what it is talking about without a tool call."""
    compounds = sorted(set(data.compound))
    source = "UNREVIEWED pipeline output (temporarily accepted)" if data.described["gold_source"] == SOURCE_PROCESSED else "Accepted videos"
    run = data.run.name if data.run else "none yet"
    return (
        f"\nData loaded: {len(data.table)} fish, {len(compounds)} compounds ({', '.join(compounds)}), vehicle is "
        f"{data.vehicle}, source: {source}, latest training run: {run}."
    )


def _is_local(host: str) -> bool:
    if host in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
