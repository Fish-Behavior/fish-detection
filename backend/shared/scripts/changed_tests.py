"""Run only the tests of the backend units whose files changed.

    python backend/shared/scripts/changed_tests.py [--base origin/master] [--all] [--list] [-- <pytest args>]

Units (backend/prepds, backend/dcs) never import each other, so a change inside one unit
needs only that unit's tests. Anything shared (backend/shared, pyproject.toml, uv.lock, Docker
and CI files) can affect every unit, so it selects all of them. Docs-only changes select nothing.
With --list it prints the selected names (prepds dcs shared) and exits; CI uses that.
"""
import argparse
import subprocess
import sys

UNITS = ("prepds", "dcs")
SHARED_FILES = {"pyproject.toml", "uv.lock", "Dockerfile", "compose.yaml", ".dockerignore", "start.sh", "stop.sh"}


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.split()


def changed_files(base: str) -> list[str]:
    fork = git("merge-base", base, "HEAD")[0]
    # committed on this branch + staged/unstaged edits + new untracked files
    return git("diff", "--name-only", fork) + git("ls-files", "--others", "--exclude-standard")


def select(files: list[str]) -> list[str]:
    names: set[str] = set()
    for f in files:
        if f.endswith(".md") or "/docs/" in f:
            continue
        if f in SHARED_FILES or f.startswith((".github/", "backend/shared/")):
            names.update((*UNITS, "shared"))
        for u in UNITS:
            if f.startswith(f"backend/{u}/"):
                names.add(u)
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/master", help="branch to compare against (default: origin/master)")
    parser.add_argument("--all", action="store_true", help="select every unit regardless of changes")
    parser.add_argument("--list", action="store_true", help="print the selected names and exit")
    args, pytest_args = parser.parse_known_args()
    names = sorted((*UNITS, "shared")) if args.all else select(changed_files(args.base))
    if args.list:
        print(" ".join(names))
        return 0
    if not names:
        print("No backend code changed; nothing to test.")
        return 0
    print("Testing:", ", ".join(names), flush=True)
    return subprocess.call([sys.executable, "-m", "pytest", *(f"backend/{n}/tests" for n in names), *pytest_args])


if __name__ == "__main__":
    sys.exit(main())
