# Training with WorkGym (the GRPO loop)

This is the concrete realization of the Sept 13 HuggingEnvs recipe: the
agent's real jobs are the RL environments, verifiable task success is the
reward, and GRPO turns rollouts into a fine-tuned model.

## The loop

```
1. Roll out each task K times (different seeds) with the current policy
       python3 examples/grpo_rollout.py --seeds 8 --out episodes/

2. Convert episodes to GRPO telemetry
       python3 -m workgym.train.grpo_adapter \
           --episodes episodes/workgym_episodes.jsonl \
           --out reports/dottie_telemetry.jsonl

3. Build preference pairs / trace bank (existing Dottie pipeline)
       cd ~/workspace/dottie/apps/ava-factory
       python3 dottie/pipeline/grpo_collect.py \
           --in /path/to/workgym/reports --out runs/grpo_pref

4. Train (Forge queue -> nugatron, per the train-gate rules:
   held-out eval win required before promoting any challenger)
```

## Why the grouping works

`grpo_collect` hashes `prompt` into `prompt_id` and computes advantages
*within* each group. The adapter emits one prompt per (env, task) —
`workgym/pair-programming-v0::retry` — so K rollouts of the same task with
different seeds form a natural GRPO group, exactly like HuggingEnvs'
multi-rollout environments.

## What the reward buys you

Every WorkGym reward is verifiable: tests pass, bugs found, labels right,
incidents diagnosed. No learned reward model, no human labeling in the
loop — the honest-reward property is what made the 4B-model result work,
and it's what makes this loop cheap to run at volume.

## Rollout policy

`examples/grpo_rollout.py` rolls out a policy over all registered envs × N
seeds using `VectorEnv`. Swap in your policy: anything that maps
`(env, obs) -> action dict`. Start with the current Dottie/Scout harness as
the policy to get a baseline, then train against it.
