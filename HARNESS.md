# HARNESS.md: agent-harness evaluation

This evaluates the **coding-agent setup** for this repo (`AGENTS.md`, its commands and gates). It is
not the product's own evals in `evals/`. Method: vibe-coding-toolkit `docs/prompts/14-harness-evaluation.md`
and `15-harness-improvement.md` (native method adaptation), Matheus Gomes, MIT, commit **790a895**
(local copy, HEAD verified). It is inspired by Harness2. It does not run the Harness2 framework,
reproduce its algorithm, or reproduce its benchmark results.

## Run on 2026-10-22 (America/Sao_Paulo)
| Step | Result | Evidence |
|---|---|---|
| Copy `templates/harness-evaluation/` to a disposable dir in /tmp (original left untouched) | done | sha256 of the 3 fixture files recorded before the copy |
| `node scripts/validate-harness.mjs <copy>` on the untouched copy | **expected baseline confirmed**: 7 tests, 5 pass, 2 fail ("denies matching user IDs across tenants", "denies missing, empty, or non-string identities") | independent command output |
| Case: boundary fix (worker fixes the fixture) | **not run** | the executing agent had no independent subagent/delegation context. Simulating success is not allowed |
| Case: stale handoff | **not run** | same reason |
| Case: read-only security review | **not run** | same reason |
| Setup reapplication / shared-resource conflict / pause-resume | **not run** | needs an independent context |
| Disposable copy removed | done | |

Client/model: Grok Bot executor (the exact model is not observable). Tokens were not observed.
No improvement trial was started: no instruction, memory or tool-use failure has been observed in
this repo yet. Prompt 15 forbids inventing one.

## Reusable procedure (run when a fresh agent context is available, or after material AGENTS.md changes)
1. `T=$(mktemp -d)`, then `cp -r <toolkit@790a895>/templates/harness-evaluation $T/fixture`. Keep the
   toolkit and this repo outside the worker's writable scope.
2. `node <toolkit>/scripts/validate-harness.mjs $T/fixture` should give 2 expected failures.
3. Dispatch one worker per case (boundary fix, stale handoff, read-only review) with the case task,
   this repo's `AGENTS.md` rules, allowed files and checks. Do not give it the fix.
4. Rerun the validator yourself, diff the files, and check that `local-notes.txt` survived. For the
   review case, compare before/after hashes.
5. Record pass/fail/not run here, separating agent self-reports from independent evidence. Then
   delete `$T`.
Never run `scripts/validate-next.mjs` (not applicable, and excluded by the owner).
