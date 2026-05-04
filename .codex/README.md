# Codex Project Configuration

This folder contains repo-local Codex defaults for working on BMAD Assist.

## Defaults

- Model: `gpt-5.5`
- Reasoning effort: `medium`
- Sandbox: `workspace-write`
- Network: enabled when the environment allows it

## Common Checks

```bash
pytest -q --tb=line --no-header
mypy src/
ruff check src/
```

## Notes

- Keep secrets in `.env.local`; it is ignored by `.gitignore`.
- Prefer narrow, command-backed changes with a clean `git status -sb` before handoff.
