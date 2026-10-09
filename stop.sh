#!/bin/sh
set -eu

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
export FISHLAB_ROOT="$PWD"
export FISHLAB_UID="$(id -u)" FISHLAB_GID="$(id -g)"
if [ -d ../fish-detection-dcs-wrapup/src/dcs ]; then
    export FISHLAB_DEFAULT_DCS_ROOT=../fish-detection-dcs-wrapup
fi

step="Docker installation"
locked=0
finish() {
    code=$?
    if [ "$code" -ne 0 ]; then
        printf '\n%s failed (exit %s). Host files remain intact.\n' "$step" "$code" >&2
    fi
    if [ "$locked" -eq 1 ]; then rmdir .fishlab-launch.lock; fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v docker >/dev/null || { echo 'Install Docker with Compose first.' >&2; exit 1; }
step="Docker Compose installation"
docker compose version >/dev/null
step="Docker daemon"
docker info >/dev/null 2>&1 || { echo 'Start Docker Desktop / the Docker daemon, then retry.' >&2; exit 1; }
step="shutdown lock"
mkdir .fishlab-launch.lock 2>/dev/null || { echo 'Another launch or shutdown is active. Check it before retrying.' >&2; exit 1; }
locked=1
cat <<'EOF'

FishLab shutdown workflow:
  1. Stop    the app (waits up to 5 min for in-flight saves)
  2. Remove  Compose containers
  3. Remove  the shutdown lock
EOF
step="Compose configuration"
docker compose config --quiet
step="App shutdown"
docker compose stop -t 300 app
step="Compose shutdown"
docker compose down -t 300
printf '\nFishLab stopped.\n'
