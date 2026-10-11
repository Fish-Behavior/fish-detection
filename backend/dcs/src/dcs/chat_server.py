"""JSON API of the research chat, for a web frontend (plan U20; D-073). Served on a loopback address only,
and only to loopback Host and Origin headers.

GET  /api/health  -> model, run, fish
GET  /api/tools   -> the query tools (OpenAI function format), e.g. to build forms or charts
POST /api/ask     -> {question, history?} -> {answer, tool_calls, history}; the frontend keeps the history
POST /api/tool    -> {name, arguments} -> the tool's JSON result, without the language model (for charts)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dcs.chat import Engine, EngineError, ask
from dcs.chat_tools import ResearchData, call_tool, tool_schemas

# The API has no authentication, so it accepts only loopback Host headers (DNS-rebinding defence) and refuses
# POSTs from a foreign Origin; the same guard as prepds' review app, restated because dcs never imports prepds.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
MAX_QUESTION = 4_000  # characters
MAX_TURN = 20_000  # characters of one earlier turn (answers can hold small tables)
MAX_TURNS = 50


class Turn(BaseModel):
    role: Literal["user", "assistant"]  # never `system` or `tool`: a client cannot rewrite the rules
    content: str = Field(max_length=MAX_TURN)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    history: list[Turn] = Field(default=[], max_length=MAX_TURNS)


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


def create_app(data: ResearchData, chat: Mapping[str, Any], engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="dcs research chat")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))
    local = {host.strip("[]") for host in ALLOWED_HOSTS}

    @app.middleware("http")
    async def refuse_foreign_origin(request: Request, call_next: Any) -> Any:
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and (urlparse(origin).hostname or "") not in local:
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": chat["model"], "run": data.run.name if data.run else None, "fish": len(data.table)}

    @app.get("/api/tools")
    def tools() -> list[dict[str, Any]]:
        return tool_schemas()

    @app.post("/api/ask")
    def ask_question(body: Question) -> dict[str, Any]:
        try:
            answer = ask(data, body.question, [turn.model_dump() for turn in body.history], chat=chat, engine=engine)
        except EngineError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        return {"answer": answer.text, "tool_calls": answer.tool_calls, "history": answer.history}

    @app.post("/api/tool")
    def run_tool(body: ToolRequest) -> dict[str, Any]:
        return call_tool(data, body.name, body.arguments)

    return app
