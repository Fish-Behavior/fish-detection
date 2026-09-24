# Contribution Workflow

Use a feature branch for every change. Do not commit or push directly to
`master`; all changes must be reviewed through a pull request (PR).

## 1. Start from the latest `master`

```bash
git switch master
git pull origin master
```

If your Git version does not support `git switch`, use `git checkout` instead.

## 2. Create a feature branch

Use a short, descriptive name with a category prefix such as `feature/`,
`fix/`, or `docs/`.

```bash
git switch -c feature/short-description
```

Examples:

```bash
git switch -c feature/video-tracking-baseline
git switch -c docs/update-contributor-guide
```

Check your current branch before making changes:

```bash
git branch --show-current
```

## 3. Make and review your changes

After editing, inspect the files and confirm that only expected changes are
included:

```bash
git status
git diff
```

Run the same checks CI runs (section 6) before committing:

```bash
pytest                                             # all tests, synthetic data only
pipx run ruff check --select F,E9 src tests        # unused imports/variables, undefined names
git ls-files | grep -Ei '\.(mp4|avi|mov|mkv|xlsx|xls|csv|pdf|parquet|npz)$|(^|/)\.env(\.|$)|^(data|outputs)/' | grep -v '\.env\.example$'  # must print nothing
```

In a notebook, clear all outputs (*Edit > Clear all outputs*) before saving.

## 4. Commit the changes

Stage only the files related to this task and use a clear commit message:

```bash
git add path/to/changed-file
git commit -m "Describe the change"
```

Avoid committing passwords, API keys, datasets, or other generated files.

## 5. Push the branch

The first push sets the upstream branch:

```bash
git push -u origin feature/short-description
```

For later commits on the same branch, use:

```bash
git push
```

## 6. Open a pull request

Create a PR from your feature branch into `master` after pushing. Include:

- a concise summary of what changed;
- the reason for the change;
- the checks or tests you ran; and
- any known limitations or follow-up work.

Request review from the relevant team members. Address review comments with
additional commits on the same branch, then push again so the PR updates.

Every push to the PR starts the checks in
[.github/workflows/ci.yml](../.github/workflows/ci.yml) (the **Checks** tab of the PR, or
the **Actions** tab). They run in parallel on GitHub's machines:

- **Privacy and lint**: no data, outputs, or `.env` in the repository; notebooks saved
  without outputs; `ruff` finds no unused or undefined names.
- **Tests (backend)**: `pytest` on Linux, macOS, and Windows, with Python 3.10 and 3.13.
- **Pages (frontend)**: the JavaScript of the live and scene-review pages parses, and the
  pages load nothing from the internet.
- **Installed package (production)**: the package is built and installed like a user
  would install it, then `all`, `live`, and `scene-review` run end to end.
- **CI passed**: green only when all of the above are.

They test the PR merged with the current `master` using read-only access, so nothing
reaches `master` before the merge. Click a failed job to see its log, fix the problem on
your branch, and push again. A newer push cancels the older run.

## 7. Keep the branch up to date

Before merging, update your branch with the latest `master` and resolve any
conflicts locally:

```bash
git fetch origin
git switch feature/short-description
git merge origin/master
git push
```

Merge only after required checks pass and the PR has been approved. A repository admin
makes the checks required once, in *Settings > Branches > Add branch protection rule*
for `master`: turn on *Require status checks to pass before merging* and select
**CI passed**. After the
PR is merged, remove the local branch if it is no longer needed:

```bash
git switch master
git pull origin master
git branch -d feature/short-description
```