# fish-detection: working rules

- Layout: `backend/prepds`, `backend/dcs`, `backend/shared` (each unit has its own src, tests, config, docs); `frontend/`; cross-unit docs in `docs/`. No root `src/`, `tests/`, `scripts/` or `config/`.
- Put new code, tests, config and docs in the unit that owns them. `prepds` and `dcs` never import each other.
- Test only what you changed: `python backend/shared/scripts/changed_tests.py` (`--all` before a PR). CI does the same per unit.
- Full rules: `docs/PRD.md` section III.3. Run steps: `docs/instructions.md`.
- Branch per change, never commit to `master` directly.
