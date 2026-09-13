# Shared project state for Codex

## Read the current baseline before acting

At the start of every task involving models, experiments, inference, comparisons,
figures or manuscript methods, re-read `CURRENT_BASELINE.json` and
`CURRENT_BASELINE.md` from this working tree. Do not rely on a previous chat's
cached baseline. These two files are the canonical current baseline registry;
dated reports, old launchers, historical Git tags and previous conversations are
historical evidence, not overrides. Explicit user instructions for a particular
experiment still take precedence.

Current registration (2026-09-13):

- ID: `mean100_v5_mean_anchor_mixed_real_no_p12_800_20260913`.
- Mean100 = **Mean-RL3 reconstruction anchor + Taylor-sqrt/Mean/Set branches +
  Gate + E3 + mean structure gradient budget <=100%**, **final step800**.
- The 100 in Mean100 is the gradient-budget setting, not 100 input frames or
  step100. Input remains 10 frames/RL3, with 90-frame training constraints.
- Canonical checkpoint, SHA256 and frozen configuration are in the JSON.
  Do not silently substitute Taylor800, old Mean400, or best740.
- Training: P01-P11 plus real fields45/55; P12 excluded. V01-V03 validation,
  T02-T04 simulation test. Both real fields are training-field diagnostics,
  have no GT, and are not independent real tests.

Before launching a new comparison, resolve the registry path and verify its
checkpoint step, anchor and SHA256. Existing frozen experiment scripts may
intentionally reference historical baselines: keep their semantics and report
incompatibility instead of bypassing their identity checks. Changing the registry
does not authorize altering another task's running jobs or starting GPU work.

Shared-file synchronization applies to tasks using this same working tree.
Already-running tasks may need an explicit request to re-read these files;
different clones/worktrees do not automatically receive uncommitted changes.
Do not claim that other tasks acknowledged the change without evidence.

## Preserve experiment provenance

Only an explicit user decision can promote a baseline. Archive the previous
registry under `docs/baseline_history/`, update both registry files and entry
notices, and record the decision and verification. Keep old weights/results/tags.
New outputs use `<experiment>_<Beijing YYYYMMDD>_runNN`, without overwriting old
results. From-scratch ablations use the recorded common initialization;
training from the current checkpoint must be labelled fine-tuning/continuation.
