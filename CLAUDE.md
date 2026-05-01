# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Guidelines

### Think before coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

### Simplicity first
- No features beyond what was asked. No abstractions for single-use code.
- No speculative flexibility or error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

### Surgical changes
- Don't "improve" adjacent code, comments, or formatting. Match existing style.
- Don't refactor things that aren't broken.
- Remove imports/variables/functions that *your* changes made unused; leave pre-existing dead code alone (mention it, don't delete it).
- Every changed line should trace directly to the user's request.

### Goal-driven execution
- Transform tasks into verifiable goals with success criteria.
- For multi-step tasks, state a brief plan with verification checks.
- Loop until verified — don't declare success without checking.

## Adversarial review

When you complete a non-trivial code change or produce a numerical result the user is likely to act on, proactively spawn a subagent (`Agent` tool — `Explore` for read-only review) for an **adversarial code review**. Assume there is a bug and hunt for sign errors, off-by-ones, unit/coordinate slips, boundary and edge-case handling, silent fallbacks, and places where an apparently safe default could bias the result. Report findings before declaring the result final.

Do this *without being asked* whenever the stakes of being wrong exceed a few minutes of agent time. Skip only for trivial edits (typo, rename, doc).

### Gold-standard rule

**Mock data is the gold standard for testing analysis pipelines.** Unit tests check the math; mocks check the *whole pipeline* — every transformation, choice, and composition step from input to output. For any calibration claim ("our pipeline recovers X to within Y"), generate synthetic input at known truth, run the pipeline end-to-end, and report bias and dispersion across multiple realizations. Single-step checks and single runs on real data can hide systematics that only surface when you ask "does the output bracket truth at the nominal level?"

## Version control

- Commit logically separable changes as distinct commits with clear messages; don't bundle unrelated work.
- Push separable issues to GitHub as separate PRs/branches so they can be reviewed and reverted independently.
- Never force-push to shared branches. Never commit secrets, large data files, or generated outputs (add to `.gitignore` if they appear).
- Before pushing, run the test suite and confirm the working tree is clean of unintended changes.
- Write commit messages that explain *why*, not just *what*. Reference issue numbers where applicable.
- If a change touches the analysis pipeline, the LaTeX writeup (see below) must be updated in the same commit or PR.

## Pipeline documentation (LaTeX)

Maintain a running LaTeX file and compiled PDF on GitHub describing the analysis pipeline and tests **as implemented by the code**. The `.tex` source and the latest compiled PDF both live in the repo and are updated whenever the pipeline changes.

After non-trivial changes to the pipeline or its tests, conduct an adversarial review (per above) whose specific goal is to verify the `.tex` text faithfully reproduces exactly what the code does — every equation, transformation, default, threshold, and test specification. Flag any drift between prose and code, and fix the prose (or the code, if the code is wrong) before declaring done.

## Plan folder

The repository contains a `docs/original-plan/` folder with the original plan and basic scripts. **Treat `docs/original-plan/` as read-only** — it is a static reference snapshot and must not be modified, even to fix typos or update outdated details.

If the plan needs to evolve as the project progresses, maintain a dynamic copy of the `.md` files at `docs/plan/`. Edit, annotate, and restructure freely there. When referencing "the plan" in discussion or commits, specify which copy (`docs/original-plan/` for the original, `docs/plan/` for the current working version) to avoid ambiguity.
