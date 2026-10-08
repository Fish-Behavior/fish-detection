FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib PATH="/opt/fishlab/.venv/bin:$PATH"
WORKDIR /opt/fishlab
COPY pyproject.toml uv.lock README.md ./
COPY src/prepds/ src/prepds/
COPY config/ config/
RUN uv sync --frozen --no-dev
COPY scripts/docker_app.py scripts/docker_app.py
COPY --from=frontend /build/dist/ frontend/dist/
ENTRYPOINT ["python", "/opt/fishlab/scripts/docker_app.py"]
CMD ["serve"]
