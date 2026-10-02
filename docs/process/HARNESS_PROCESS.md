# Harness engineering process

A repeatable way to take any project from idea to reviewed, documented code with an AI coding agent. Copy this file to a new repository unchanged; everything specific to one project goes in that project's PRD.

Companion file: [PRD_TEMPLATE.md](PRD_TEMPLATE.md).

## 1. Tools and what they do

Checked by reading the two repositories, not from memory. "To check" items need a look on your own machine.

| Tool | What it is | Verified facts | To check on your machine |
|---|---|---|---|
| **Claude Code** | The coding agent (terminal, VS Code, web) | - | - |
| **harness-os** ([repo](https://github.com/lehoangbaoduy/harness_os)) | A governance MCP server that runs in Docker with a Postgres database | Tools: `get_constitution`, `create_spec`, `validate_spec`, `assess_risk`, `generate_tests`, `request_review`, `run_workflow`, `workflow_status`, `record_decision`, `trace_artifact`, `impact_analysis`, `validate_coverage`, `audit_report`, `evaluate_governance`. Workflows: `new-feature` (spec → risk → tests → implement → review → finalize → trace), `hotfix`, `security-change`. Rules (the "constitution"): no gated code without a validated spec; a test must be observed failing (RED) before the code and passing (GREEN) after; config files are integrity-checked; agent boundaries. A rule-based risk rubric sets low/medium/high/critical, and critical needs a human approval at a terminal | Containers `harness_postgres` and `harness_gate_daemon` are running; the MCP server is registered in `~/.claude.json` |
| **Everything Claude Code (ECC)** ([fork](https://github.com/WorldFlowAI/everything-claude-code)) | A Claude Code plugin of agents, slash commands, skills, rules and hooks | This fork is a January 2026 snapshot of the upstream project. It contains agents (`planner`, `tdd-guide`, `code-reviewer`, `security-reviewer`, `architect`, `e2e-runner`, `doc-updater`, `refactor-cleaner`, `build-error-resolver`) and commands (`/plan`, `/tdd`, `/verify`, `/code-review`, `/checkpoint`, `/e2e`, `/eval`, `/learn`, `/update-docs`, …) | Which version you installed: harness-os calls ECC skills (`orch-add-feature`, `orch-fix-defect`) and reviewers (`python-reviewer`, `fastapi-reviewer`, `typescript-reviewer`, `database-reviewer`) that are **not** in this fork's snapshot. Check that your installed ECC has them |
| **Playwright MCP** | Drives a real browser so the agent sees what actually renders | - | Registered as an MCP server; used for every frontend or full-stack change |

## 2. File and naming convention

All documents live in `docs/`. A subfolder is used when several files share a purpose.

| File | Purpose |
|---|---|
| `docs/<project>_PRD.md` | Source of truth: what, why, data, evaluation, acceptance checklist. A repository with a single project may simply use `docs/PRD.md`; as soon as a second project exists, prefix every file with its project name so nothing collides |
| `docs/plans/<project>_plan.md` | Implementation plan derived from the PRD |
| `docs/<project>_progress.md` | Live status and design decisions, updated after every step |
| `docs/process/` | This process and the PRD template |

## 3. The process

1. **Interview and PRD.** The agent asks the questions in §6, then drafts `docs/<project>_PRD.md` from the template. Remove sections that do not apply. The agent never guesses: unclear points go to the PRD's open-questions table with a default.
2. **Privacy gate.** Before any text leaves the machine (Git, another tool, a chat), check it for real names, IDs, dates, paths and data. Use placeholders.
3. **Review the PRD yourself.** Nothing is implemented until you have read and approved it.
4. **Plan.** In a fresh session: read the PRD, explore the codebase, ask clarifying questions about requirements, constraints and edge cases, then write `docs/plans/<project>_plan.md`. Phases come from PRD §10, tests from PRD §9, "done" from PRD §1.6 and §9.3.
5. **Challenge the plan.** A second session (a different model only if the privacy gate allows) looks for holes. Revise and note what changed.
6. **Implement in small steps with a verification loop:**
   1. write the test or execution script first;
   2. implement the step;
   3. run the loop and compare with the expected output;
   4. if it differs, debug before moving on;
   5. update the progress file and record every design decision, especially where the PRD left a detail open.
7. **Review.** Review the uncommitted changes against the plan and the PRD like a staff engineer: correctness, performance, security, privacy. A second reviewer is preferred.
8. **Finalize.** Update the plan and progress file with what was actually built and why; that is the documentation. Commit only when the owner says so.

### The harness pipeline inside step 6

When harness-os governance is active for the project, each unit of work follows its order: **Constitution → Specification → Test Suite → Generate Code → Code Review**, run through `run_workflow` (`new-feature`, `hotfix` or `security-change`). The server advances a stage only when it observes evidence (for example a test run observed failing).

## 4. Before turning harness-os governance on for a project

`harness init` is a one-time, per-project step. It is not "just configuration", so decide it deliberately:

- It writes `.claude/` (hooks, `settings.json`, `harness.config.json`, `agents/AGENTS.md`) **and makes a git commit** ("harness: initial project scaffold").
- For Python it gates every `**/*.py` file: writes are blocked until a validated spec covers the file and its test has been seen failing.
- Its gate daemon can only see the folder mounted as `HARNESS_PROJECTS_ROOT`, so the project must live under it.
- Its runbook uses the author's own paths (`/home/lehoa/...`) and a WSL2 + Docker setup; adapt them to your machine.
- Risk is classified by keywords in the request text and paths. Words such as `order`, `token`, `session` or `credential` can raise the level and trigger a human approval prompt even in harmless work; expect to see this and read the rationale rather than ignoring it.
- Config files are integrity-checked; if you edit one deliberately, run `harness reconcile-config <project>`.

Ask the owner before running it, and say plainly that it commits.

## 4b. Repository conventions to record per project

- **Tests:** keep them in `tests/`. Related tests may be grouped in subfolders (for example `tests/<package>/`). Test files need unique base names across folders unless each folder has an `__init__.py`; shared fixtures go in a `conftest.py` in the subfolder. Check that the CI runs the subfolder (a recursive `pytest tests/` does).

## 5. Verification loop by project type

| Project part | Loop |
|---|---|
| Backend / library | Linter, unit tests, an end-to-end run on synthetic data |
| Frontend / full-stack | **Playwright MCP**: open the page, look at the rendering, check the console and the key flows. Do not infer correctness from code |
| Data / ML (add these) | Leakage tests (no row or group in two folds; scalers fit on training data only), fixed seeds, forbidden-column assertions, baselines reported next to every model, a reload-and-reproduce test for saved models |

Set these loops up before the agent starts implementing, and run them after every significant change, not only at the end of a phase.

## 6. Questions the agent asks at the start of every project

**Goal and users**
1. What problem is solved, for whom, and what is the cost of not solving it?
2. What does "done" mean, and who decides?
3. What is explicitly out of scope?

**Data and privacy**
4. What data exists, how much, and what is restricted? What may never appear in Git, chats or documents?
5. Where may data be stored and moved (approved cloud storage, USB, direct copy)?
6. Which names, IDs and dates need placeholders?

**Technical**
7. Language, frameworks, and what must keep working (operating systems, Python versions, CI)?
8. Where does it run (local, remote machine, GPU), and how is it reached and monitored?
9. How are results and models retrieved and re-used?

**Quality**
10. Which risks matter most (leakage, security, correctness, performance)?
11. What are the known edge cases and failure modes?
12. What must be tested before code is written?

**Process**
13. Is harness-os governance on for this project (see §4)? Which verification tool for the UI?
14. May the agent commit and push, or only when told?
15. Are external review tools allowed, and on what material?
16. Deadline, reviewers, and the gates where you want to stop and look?

## 7. Master prompt (fill the brackets per project)

```
I want to implement [project/feature]. The full PRD is at docs/[project]_PRD.md; read it in full first.
It is the source of truth; this prompt sets the process, not the spec.

1. Plan: write docs/plans/[project]_plan.md. Take the phases from PRD §10, the tests from
   PRD §9 (every row of the edge-case catalog is a required task), and "done" from PRD §1.6
   and §9.3. Ask me clarifying questions before writing it.
2. Implement phase by phase. For each step: implement, run the verification loop, confirm the
   output matches, debug before moving on, and record progress and design decisions in
   docs/[project]_progress.md.
3. Backend and ML: write execution scripts or tests first; run them after every significant change.
4. Frontend: use the Playwright MCP server and look at what renders.
5. If harness-os governance is on, follow its pipeline for every unit of work and do not skip stages.
6. Review the uncommitted changes against the plan and the PRD for correctness, performance,
   security and privacy before calling a step done.
7. Do not commit, push or open a pull request unless I ask. Ask me when anything is unclear.

Don't miss these (project-specific): [easy-to-forget requirements]
```

Keep the progress file current throughout; it must always show the real status, not a snapshot from when the plan was written.

## 8. Lessons from the first project (add to this list as projects finish)

1. **Look at the real data before designing validation.** A one-minute check of how classes, dates and batches overlap decided the whole evaluation design of the first project.
2. **Check upstream readiness first.** Ask what has actually been run, reviewed and accepted before planning work that depends on it.
3. **Confirm the repository layout before writing paths.** Folders were moved and a package was retired while the PRD was being written; a quick `git fetch` and a look at the default branch avoids stale references.
4. **Name files by project from the start** (§2), because repositories grow a second project.
5. **Say what is infrastructure and what is a claim.** If the inputs are not yet validated, the PRD must say that results are not scientific claims.
6. **Record who answered what.** Keep the open-questions table with status and default, so a new session can continue without re-asking.
