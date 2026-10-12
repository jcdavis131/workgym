# WorkGym

Gymnasium-style RL environments for **real work workflows** — not games.
Stdlib only. Zero third-party dependencies.

Built for the HuggingEnvs-style GRPO loop: the agent's real jobs become
environments, and verifiable task success becomes the reward.

## Quickstart

```python
import workgym

print(workgym.list_envs())
# ['workgym/inbox-triage-v0', 'workgym/pair-programming-v0', 'workgym/pr-review-v0']

env = workgym.make("workgym/pair-programming-v0")
obs, info = env.reset(seed=7)          # obs: task brief + starter files
obs, reward, term, trunc, info = env.step(
    {"op": "write_file", "path": "solution.py", "content": "..."})
obs, reward, term, trunc, info = env.step({"op": "run_tests"})   # dev loop, reward 0
obs, reward, term, trunc, info = env.step({"op": "submit"})      # reward = pass rate
```

## Environments

| id | episode | reward |
|---|---|---|
| `workgym/pair-programming-v0` | 6 coding tasks (retry, LRU, deep-merge, duration parse, top-k, sliding median) in a temp sandbox; `write_file` / `run_tests` / `submit` | fraction of hidden tests passing |
| `workgym/pr-review-v0` | 5 diffs (4 with seeded security/correctness/perf bugs, 1 clean); submit findings | F1 on (file, line) bug detection, ±2 lines |
| `workgym/inbox-triage-v0` | 3 inboxes × 10 emails; label URGENT/ACTION/FYI/NOISE | label accuracy |
| `workgym/cron-ops-v0` | 5 ops incidents (dead trigger, stale feed, port conflict, RAM pressure, silenced cron); pick root cause + fix from candidates | 0.5 + 0.5 |

Rewards are always in [0, 1] and always verifiable. The pr-review bugs are
seeded by design (training env, not a published benchmark) — the point is the
reward is exact, not vibes.

## API

- `workgym.make(env_id, **kwargs)`, `workgym.register(...)`, `workgym.list_envs()`
- `reset(seed) -> (obs, info)` · `step(action) -> (obs, reward, terminated, truncated, info)`
- `workgym.vec.VectorEnv([factory, ...])` — thread-parallel batched envs
- `workgym.log.EpisodeLogger` — append-only JSONL (`WORKGYM_LOG_DIR` or
  `~/workspace/workgym/episodes/`), shaped to feed the Dottie `grpo_collect`
  pipeline: `{env_id, episode_id, seed, steps[], total_reward, info}`

## Training lane

See [TRAINING.md](TRAINING.md): roll out with `examples/grpo_rollout.py`,
convert with `workgym.train.grpo_adapter`, feed the existing Dottie
`grpo_collect` pipeline. Same prompt → same `prompt_id`, so K rollouts of
one task form a GRPO group.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

14 tests, all green. The load-bearing ones: reference solutions score **1.0**
on all 6 coding tasks, gold findings score **1.0** on all 5 diffs, gold labels
score **1.0** on all 3 inboxes. If a task's tests can't be solved by their own
reference, the suite fails — the envs can't drift from solvable.

## Honest limitations (v1)

- The pair-programming sandbox is a temp dir, not a security boundary — don't
  run untrusted agent code here without a container.
- Hidden tests live on disk in the sandbox (excluded from observations, but a
  sufficiently curious agent could read them — fine for training, not for eval).
- Single-step review/triage envs; multi-step variants (ask-for-context,
  draft-then-send) are the obvious v2.
- No pixel/game envs, on purpose.

## Roadmap

- GRPO trainer hookup (the Sept 13 HuggingEnvs recipe)
- `dottie-pair-live-v0`: traces from real pair sessions as tasks
- `cron-ops-v0`: diagnose a failing cron/hook from logs
- Human-preference env: accept/reject signals as reward
