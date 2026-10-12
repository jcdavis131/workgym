# Changelog

All notable changes to WorkGym. Format: Keep a Changelog.

## [0.1.0] - 2026-10-12

First production-grade release.

### Added
- Gymnasium-compatible core (`Env`, `Text`/`Discrete`/`Dict` spaces, `make`/`register`/`list_envs`) — stdlib only, zero dependencies
- `workgym/pair-programming-v0`: 6 coding tasks in an isolated sandbox (`write_file`/`run_tests`/`submit`); reward = hidden-test pass rate
- `workgym/pr-review-v0`: 5 diffs (4 seeded bugs + 1 clean); reward = F1 on (file, line) detection ±2 lines
- `workgym/inbox-triage-v0`: 3 inboxes × 10 emails, URGENT/ACTION/FYI/NOISE rubric; reward = label accuracy
- `workgym/cron-ops-v0`: 5 ops incidents modeled on real fleet history; reward = 0.5 root cause + 0.5 fix
- `VectorEnv`: thread-parallel batched envs
- `EpisodeLogger`: append-only JSONL shaped for the Dottie `grpo_collect` pipeline
- `workgym.train.grpo_adapter`: episodes → GRPO telemetry (`prompt_id` groups K seeds per task)
- `examples/grpo_rollout.py`: rollout driver over all envs × N seeds
- `TRAINING.md`: the full GRPO loop documented
- CI (GitHub Actions, Python 3.10–3.12): unit tests + rollout smoke + adapter smoke
- Determinism tests: same seed → same task, reproducible rewards

### Guarantees
- Rewards always in [0, 1], always verifiable — never vibes
- Reference/gold solutions score exactly 1.0 on every task (suite fails otherwise)
- Same seed → same episode, or it's a bug

### Known limits
- Pair-programming sandbox is a temp dir, not a security boundary
- Hidden tests live on disk (excluded from obs; fine for training, not for held-out eval)
- Single-step review/triage/ops envs in v1
