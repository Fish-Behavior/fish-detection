"""JSON API of the research chat, for a web frontend (plan U20; D-073). Served on a loopback address only.

GET  /api/health  -> model, run, fish
GET  /api/tools   -> the query tools (OpenAI function format), e.g. to build forms or charts
POST /api/ask     -> {question, history?} -> {answer, tool_calls, history}; the frontend keeps the history
POST /api/tool    -> {name, arguments} -> the tool's JSON result, without the language model (for charts)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dcs.chat import Engine, EngineError, ask
from dcs.chat_tools import ResearchData, call_tool, tool_schemas


class Question(BaseModel):
    question: str
    history: list[dict[str, str]] = []


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


def create_app(data: ResearchData, chat: Mapping[str, Any], engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="dcs research chat")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": chat["model"], "run": data.run.name if data.run else None, "fish": len(data.table)}

    @app.get("/api/tools")
    def tools() -> list[dict[str, Any]]:
        return tool_schemas()

    @app.post("/api/ask")
    def ask_question(body: Question) -> dict[str, Any]:
        try:
            answer = ask(data, body.question, body.history, chat=chat, engine=engine)
        except EngineError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        return {"answer": answer.text, "tool_calls": answer.tool_calls, "history": answer.history}

    @app.post("/api/tool")
    def run_tool(body: ToolRequest) -> dict[str, Any]:
        return call_tool(data, body.name, body.arguments)

    return app
