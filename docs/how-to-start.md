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

Run the project checks described in the repository documentation before
committing.

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

## 7. Keep the branch up to date

Before merging, update your branch with the latest `master` and resolve any
conflicts locally:

```bash
git fetch origin
git switch feature/short-description
git merge origin/master
git push
```

Merge only after required checks pass and the PR has been approved. After the
PR is merged, remove the local branch if it is no longer needed:

```bash
git switch master
git pull origin master
git branch -d feature/short-description
```