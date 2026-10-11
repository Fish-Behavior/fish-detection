#!/bin/sh
set -eu

full=0
setup=0
for arg in "$@"; do
    case "$arg" in
        --setup) setup=1 ;;
        --full) full=1; setup=1 ;;
        *) echo 'Usage: ./start.sh [--setup | --full]
  (no flag)  build if needed, start the app; needs a finished setup
  --setup    build + prepare data (check, catalog, track, DCS) when needed, then stop
  --full     like --setup but redo every step' >&2; exit 2 ;;
    esac
done

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
Two modes: ./start.sh --setup does steps 1-5 (slow, once per machine or after inputs change);
plain ./start.sh builds if needed and runs step 7, and tells you to run --setup when it is required.
Fast path: if the image and the inputs (.env, workbook, videos, config, DCS checkout) are unchanged
since the last successful run, steps 1-5 are skipped and only step 7 runs. Use ./start.sh --full to redo everything.
Review and accept videos in the browser, then run ./stop.sh to shut down.

Average waiting times (steps under ~1 min are not listed):
  - 1 Build:  a few minutes the first time; later runs reuse the cache
  - 3 Track:  ~35 s per 20-min video per worker; all 328 videos took ~50 min on 22 workers
              (set PDS_WORKERS in .env). Videos already processed are skipped, so reruns are fast.
  - 5 DCS:    only the model training is slow and only when no saved model exists; not timed

EOF
# Fast path: stamps record the last fully successful build / preparation. When nothing they depend on
# changed, those steps are skipped. Any doubt (missing stamp, newer file, different video count,
# failed previous prepare, --full) falls back to the full, safe path.
build_stamp=.fishlab-build.stamp
ready_stamp=.fishlab-ready.stamp
changed() { # changed STAMP PATH... : true if STAMP is missing or any existing PATH has a file newer than it
    stamp=$1; shift
    [ -f "$stamp" ] || return 0
    for path in "$@"; do
        [ -e "$path" ] || continue
        [ -n "$(find "$path" -newer "$stamp" -type f ! -path '*/node_modules/*' ! -path '*/dist/*' -print -quit 2>/dev/null)" ] && return 0
    done
    return 1
}
# Only .mp4 files matter to the catalog. PDS_VIDEO_DIR may be the repo itself, so skip generated/work folders.
mp4s() { find "$(envval PDS_VIDEO_DIR)" \( -name .git -o -name .venv -o -name node_modules -o -name outputs -o -name accepted \) -prune -o -type f -iname '*.mp4' "$@" -print 2>/dev/null; }
video_count() { mp4s | wc -l | tr -d ' '; }
videos_changed() { [ -f "$ready_stamp" ] || return 0; [ -n "$(mp4s -newer "$ready_stamp" | head -n 1)" ]; }
output_dir="$(envval PDS_OUTPUT_DIR)"; output_dir="${output_dir:-outputs}"

need_build=1
if [ "$full" -eq 0 ] && docker image inspect fishlab:local >/dev/null 2>&1 \
    && ! changed "$build_stamp" Dockerfile pyproject.toml uv.lock README.md src/prepds config scripts/docker_app.py frontend; then
    need_build=0
fi

need_prepare=1
if [ "$full" -eq 0 ] && [ -f "$output_dir/docker/startup.json" ] \
    && grep -q '"ok": true' "$output_dir/docker/startup.json" \
    && [ "$(cat "$ready_stamp" 2>/dev/null)" = "$(video_count)" ] \
    && ! videos_changed \
    && ! changed "$ready_stamp" .env "$(envval PDS_DB_PATH)" config \
        "${FISHLAB_DCS_ROOT:-${FISHLAB_DEFAULT_DCS_ROOT:-}}"; then
    need_prepare=0
fi
# Setup also redoes preparation after an image rebuild (the code that produces the outputs changed).
if [ "$setup" -eq 1 ] && [ "$need_build" -eq 1 ]; then need_prepare=1; fi
if [ "$setup" -eq 0 ] && [ "$need_prepare" -eq 1 ]; then
    step="setup check"
    printf '\nFishLab is not set up yet, or its inputs changed (.env, workbook, config, videos, DCS).\nRun  ./start.sh --setup  first (builds and prepares the data), then ./start.sh.\nNothing was started.\n' >&2
    exit 1
fi

if [ "$need_build" -eq 1 ]; then
    step="frontend and Python image build"
    docker compose build
    touch "$build_stamp"
else
    echo '[SKIP] Build: image is up to date.'
fi
if [ "$need_prepare" -eq 1 ]; then
    rm -f "$ready_stamp"
    step="stop the previous Compose app"
    docker compose stop app
    step="prepds / DCS preparation (see the named step in the log)"
    docker compose up --no-build --force-recreate --abort-on-container-exit --exit-code-from prepare prepare
    video_count > "$ready_stamp"
else
    echo '[SKIP] Check, Track, Verify, DCS: nothing changed since the last successful preparation (use --full to redo).'
fi
if [ "$setup" -eq 1 ]; then
    diagnostics=0
    printf '\nSetup finished. Run ./start.sh to open FishLab.\n'
    exit 0
fi
step="frontend + backend startup and health check"
docker compose up -d --no-build --no-deps --wait --wait-timeout 60 app
step="master link"
address="$FISHLAB_HOST:$FISHLAB_PORT"
printf '\nFishLab ready: http://%s\nStop: ./stop.sh\nLogs: docker compose logs -f app\n' "$address"
