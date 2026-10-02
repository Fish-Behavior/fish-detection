"""Guards shared by the local review and annotation apps: they have no authentication, so they accept only
loopback Host headers (DNS-rebinding defence) and refuse state-changing requests with a foreign Origin."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def install_local_guards(app: FastAPI, allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS) -> None:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    hosts = {h.strip("[]") for h in allowed_hosts}

    @app.middleware("http")
    async def refuse_foreign_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in _UNSAFE_METHODS and origin and (urlparse(origin).hostname or "") not in hosts:
            return JSONResponse(
                {"detail": {"code": "foreign_origin", "message": "cross-origin request refused"}}, status_code=403
            )
        return await call_next(request)
