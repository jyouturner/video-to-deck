---
name: bump-and-ship
description: Commit + push the current branch using a tempfile-based commit message (so apostrophes / em-dashes / other characters that break heredoc quoting are safe). Stages files by name only; never uses `git add -A`.
---

# bump-and-ship

Use this when the user wants to commit and push. Skip if no actual commit is requested.

## Why a skill instead of just doing it

We hit the same heredoc apostrophe bug multiple times before this skill existed: writing a commit message inline with `git commit -m "$(cat <<'EOF' ... EOF)"` breaks when the body contains `'` (e.g. "Claude's", "the LLM's"), because Bash's quote-parser treats that as the heredoc terminator. The fix is always the same: write the message to a tempfile, then `git commit -F <tempfile>`. This skill bundles that procedure so we don't re-derive it.

## Inputs

Before invoking, you need (from the user or from your prior reasoning):

1. **A subject line** — 1 short sentence, under ~70 chars, matching the existing commit style in this repo (e.g. *"Topic: short imperative description"*). Look at `git log --oneline -10` to copy the style.
2. **A body** — 1–3 paragraphs explaining the *why*, not the *what*. The diff already shows the what.
3. **The files to stage** — explicit list. NEVER `git add -A` / `git add .` per the system-prompt safety rules (sensitive files, large binaries).

If any of these are missing, ask the user before staging.

## Procedure

1. **Show context**. Run `git status` and `git diff --stat` in parallel so the user (and you) can see the scope before committing.

2. **Pre-flight checks**:
   - Working tree must have actual changes (no empty commits).
   - Files about to be staged are not `.env`, credentials, or large binaries. If any of those appear in `git status`, stop and warn.
   - Branch is not detached HEAD.

3. **Write the commit message** to `/tmp/yt2md-commit-msg.txt` using the Write tool. Format:

   ```
   <subject>

   <body paragraph 1>

   <body paragraph 2>

   Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
   ```

   The blank line between subject and body is required for git to parse them correctly.

4. **Stage + commit + push** in a single Bash command (saves a turn):

   ```bash
   git add <file1> <file2> ... && \
     git commit -F /tmp/yt2md-commit-msg.txt && \
     git push origin <current-branch> && \
     git log --oneline -3
   ```

   The final `git log` is the receipt showing the commit landed.

5. **If anything fails**:
   - **Pre-commit hook failed**: investigate the failure, fix the underlying issue, re-stage, create a **NEW** commit. Never use `--amend` after a hook failure — the original commit didn't happen, so amend would modify the *previous* commit and lose history.
   - **Push rejected**: surface the rejection to the user. Do NOT force-push (especially not to `main`).
   - **Dirty files remain after commit**: tell the user; ask if those belong in a follow-up commit or were intentionally left out.

## Branch handling

- Default push target is `origin <current-branch>`. Get the current branch with `git branch --show-current`.
- If pushing to `main`/`master`, double-check this is intentional for the change (small fixes, yes; experimental rewrites, no).

## What this skill is NOT for

- Drafting prose for the commit message — that's your job; this just handles the mechanics.
- Resolving merge conflicts — use the standard git flow for that.
- Force-pushes — never automated; if you genuinely need one, ask the user first.

## Standard footer

Every commit body ends with exactly this line (do not change the model name):

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
