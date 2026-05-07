# CLAUDE.md
 
Guidance for Claude Code (claude.ai/code) when working with code in this repository.
 
## Behavioral Guidelines
 
### Think before coding
- State assumptions explicitly; ask if uncertain.
- If multiple interpretations exist, present them — don't pick silently.
- Push back when a simpler approach exists.
### Simplicity first
- No features, abstractions, or error handling beyond what was asked.
- If 200 lines could be 50, rewrite it.
### Surgical changes
- Match existing style. Don't refactor or "improve" adjacent code.
- Remove imports/variables your changes made unused; leave pre-existing dead code alone (mention it, don't delete it).
- Every changed line should trace to the user's request.
### Goal-driven execution
- Transform tasks into verifiable goals with success criteria.
- For multi-step work, state a brief plan with verification checks.
- Loop until verified — don't declare success without checking.

## Working style and routing
 
Default to the main session for small edits, tight back-and-forth, and final judgment. Delegate to subagents when output would be verbose, work is parallelizable, or a task is self-contained.
 
Subagents:
 
- **worker** (Sonnet): implements specified changes after the approach is decided.
- **reviewer** (Sonnet): default review pass after non-trivial worker output.
- **deep-reviewer** (Opus): security, concurrency, auth, data integrity, public APIs, or anything flagged critical.
- **analyst** (Opus): fresh-context reasoning when the main thread is noisy or the question is orthogonal. Read-only.
- **test-runner** (Haiku): runs tests, returns only failures.
- **researcher** (Haiku): external docs, library APIs, web lookup.
- **Explore** (built-in, Haiku): read-only codebase lookup. Used automatically.
Rules:
 
1. After non-trivial worker output, invoke reviewer (or deep-reviewer for high-stakes code) before declaring done.
2. After worker completes, invoke test-runner unless the change is purely cosmetic.
3. Fan out parallel subagents for independent investigations across multiple files.
4. Never run tests inline — delegate to test-runner so logs stay out of context.
5. Resolve ambiguity, wide design space, or subagent questions in the main session before (re-)dispatching. Worker does not make architectural decisions.

## Adversarial review
 
After any non-trivial code change or numerical result the user is likely to act on, run an **adversarial code review**. Assume there is a bug; hunt for sign errors, off-by-ones, unit/coordinate slips, boundary handling, silent fallbacks, and biased defaults. Report findings before declaring final.
 
Routing: **reviewer** for routine passes; **deep-reviewer** when the change touches the analysis pipeline, numerical methods, calibration, or anything where a subtle bug propagates into results. Do this *unprompted* whenever the cost of being wrong exceeds a few minutes of agent time. Skip only for trivial edits (typo, rename, doc).
 
Reviewers consult `docs/review-checklist.md` for recurring bug classes in this repo. When adversarial review (or a user) catches a bug whose class isn't already listed, append it.
 
### When the gate fires
 
The gate fires **before each commit** that touches non-trivial code — three commits = three reviewer dispatches. The diff under review is the diff being committed.
 
### What does NOT substitute for a reviewer pass
 
- **Parity gates / regression numbers / unit tests passing.** These verify what they were aimed at; a reviewer finds what you didn't think to check. Parity can't catch bugs shared by both paths or in code outside the comparison surface.
- **"Framework-only, no calibration."** Import order, registry dispatch, docstring claims, and silent fallbacks are reviewable even when no numerics change.
- **The user already pointed out a bug.** That raises the prior of more bugs; the fix itself needs scrutiny for edge cases and adjacent interactions.
- **"Small extension of already-reviewed code."** Non-trivial new code gets its own pass.

### Gold-standard rule
 
**Mock data is the gold standard for testing analysis pipelines.** Unit tests check the math; mocks check the *whole pipeline* end-to-end. For any calibration claim ("recovers X to within Y"), generate synthetic input at known truth, run the pipeline, and report bias and dispersion across multiple realizations. Single-step checks and single runs on real data hide systematics.
 
Delegate mock runs: **worker** to set up and execute, **test-runner** for the runs, **deep-reviewer** to audit the bias/dispersion claim before it enters the writeup.
 
## Version control
 
- Commit logically separable changes as distinct commits; don't bundle unrelated work.
- Push separable issues as separate PRs so they can be reviewed and reverted independently.
- Never force-push to shared branches. Never commit secrets, large data files, or generated outputs (add to `.gitignore`).
- Before pushing: run tests via test-runner and confirm the working tree is clean.
- Commit messages explain *why*, not just *what*. Reference issue numbers where applicable.
- Pipeline changes and the LaTeX writeup (below) must land in the same commit or PR.
## Pipeline documentation (LaTeX)
 
Maintain a running `.tex` source and compiled PDF in the repo describing the pipeline and tests **as implemented**. Update whenever the pipeline changes.
 
After non-trivial pipeline or test changes, run an adversarial review (use **deep-reviewer**) to verify the `.tex` faithfully reproduces what the code does — every equation, transformation, default, threshold, and test spec. Prose-vs-code drift is a high-stakes failure mode. Fix the prose (or the code, if the code is wrong) before declaring done.
 
## Plan folder
 
`docs/original-plan/` is **read-only** — a static reference snapshot. Do not modify it, even for typos.
 
For an evolving plan, maintain a working copy at `docs/plan/`. When referencing "the plan," specify which copy to avoid ambiguity.