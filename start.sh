#!/bin/sh
set -eu

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
export FISHLAB_ROOT="$PWD"
export FISHLAB_UID="$(id -u)" FISHLAB_GID="$(id -g)"
if [ -d ../fish-detection-dcs-wrapup/src/dcs ]; then
    export FISHLAB_DEFAULT_DCS_ROOT=../fish-detection-dcs-wrapup
fi

step="Docker installation"
diagnostics=0
locked=0
fail() {
    code=$?
    if [ "$code" -ne 0 ]; then
        printf '\nFAILED: %s (exit %s). FishLab was not started successfully.\n' "$step" "$code" >&2
        if [ "$diagnostics" -eq 1 ]; then
            docker compose ps -a >&2 || true
            docker compose logs --tail 60 prepare app >&2 || true
            if [ "$step" = "frontend + backend startup and health check" ]; then
                docker compose stop app >&2 || true
            fi
        fi
    fi
    if [ "$locked" -eq 1 ]; then rmdir .fishlab-launch.lock; fi
}
trap fail EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v docker >/dev/null || { echo 'Install Docker with Compose first.' >&2; exit 1; }
step="Docker Compose installation"
docker compose version >/dev/null
step="Docker daemon"
docker info >/dev/null 2>&1 || { echo 'Start Docker Desktop / the Docker daemon, then retry.' >&2; exit 1; }
step=".env configuration"
# Setup check: report everything that is not configured yet, then stop before any slow work.
# Last KEY=value in .env, without CR, surrounding quotes, inline " # comment" or spaces.
envval() { sed -n "s/^$1=//p" .env 2>/dev/null | tail -n 1 | tr -d '\r' | sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^[\"']\(.*\)[\"']\$/\\1/"; }
missing=""
if [ ! -f .env ]; then
    missing="
  - .env is missing: run  cp .env.example .env  then edit it."
else
    [ -d "$(envval PDS_VIDEO_DIR)" ] || missing="$missing
  - PDS_VIDEO_DIR is unset or not a folder (the synced video mirror)."
    [ "$(envval FISHLAB_HOST)" = 127.0.0.1 ] || missing="$missing
  - FISHLAB_HOST must be 127.0.0.1 (Docker needs a loopback IP; the app has no authentication)."
    case "$(envval FISHLAB_PORT)" in ''|*[!0-9]*) missing="$missing
  - FISHLAB_PORT is unset or not a number (the local port for the app link).";; esac
    [ -f "$(envval PDS_DB_PATH)" ] || missing="$missing
  - PDS_DB_PATH is unset or not a file (the trial database workbook)."
fi
if [ -n "$missing" ]; then
    printf 'FishLab is not configured yet. Please set up:%s\n\nThen run ./start.sh again. Nothing was started.\n' "$missing" >&2
    exit 1
fi
# Compose publishes these exact values, so the printed link always matches .env.
export FISHLAB_HOST="$(envval FISHLAB_HOST)" FISHLAB_PORT="$(envval FISHLAB_PORT)"
step="startup lock"
mkdir .fishlab-launch.lock 2>/dev/null || { echo 'Another launch is active. If it was interrupted with SIGKILL, remove .fishlab-launch.lock after verifying no launch is running.' >&2; exit 1; }
locked=1
step="Compose configuration"
docker compose config --quiet
diagnostics=1
cat <<'EOF'

FishLab startup workflow (stops at the first failed step and names it):
  1. Build      Docker image with the frontend and Python code
  2. Check      prepds check-config, then catalog (match trial rows to videos)
  3. Track      prepds run: track + label every video not yet processed
  4. Verify     every processed video has its output files
  5. DCS        featurize, check-config, audit, model (trains only if none is saved), predict
  6. Chat       optional, only if enabled in .env
  7. Serve      start frontend + API, wait for health, print the link
Review and accept videos in the browser, then run ./stop.sh to shut down.

Average waiting times (steps under ~1 min are not listed):
  - 1 Build:  a few minutes the first time; later runs reuse the cache
  - 3 Track:  ~35 s per 20-min video per worker; all 328 videos took ~50 min on 22 workers
              (set PDS_WORKERS in .env). Videos already processed are skipped, so reruns are fast.
  - 5 DCS:    only the model training is slow and only when no saved model exists; not timed

EOF
step="frontend and Python image build"
docker compose build
step="stop the previous Compose app"
docker compose stop app
step="prepds / DCS preparation (see the named step in the log)"
docker compose up --no-build --force-recreate --abort-on-container-exit --exit-code-from prepare prepare
step="frontend + backend startup and health check"
docker compose up -d --no-build --no-deps --wait --wait-timeout 60 app
step="master link"
address="$FISHLAB_HOST:$FISHLAB_PORT"
printf '\nFishLab ready: http://%s\nStop: ./stop.sh\nLogs: docker compose logs -f app\n' "$address"
