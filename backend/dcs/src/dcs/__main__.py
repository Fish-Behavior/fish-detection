"""Allows `python -m dcs <command>` (same as the `dcs` command)."""

from dcs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
