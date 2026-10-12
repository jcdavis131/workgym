"""GRPO rollout driver: every registered env x N seeds -> episode JSONL.

Usage:
    python3 examples/grpo_rollout.py --seeds 4
    python3 examples/grpo_rollout.py --seeds 8 --env workgym/pair-programming-v0

POLICY is the placeholder — swap it for your agent. Anything that maps
(obs, rng) -> action dict. The point of this script is the loop mechanics:
VectorEnv rollouts, seeded episodes, JSONL out, ready for grpo_adapter.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workgym
from workgym.log import EpisodeLogger
from workgym.vec import VectorEnv


def placeholder_policy(env_id):
    """Returns obs,rng -> action. Deliberately dumb; replace with your agent."""
    def pair_programming(obs, rng):
        return {"op": "submit"}

    def pr_review(obs, rng):
        return {"op": "submit", "findings": []}

    def inbox_triage(obs, rng):
        return {"op": "submit",
                "labels": {e["id"]: "NOISE" for e in obs["emails"]}}

    def cron_ops(obs, rng):
        return {"op": "submit",
                "root_cause": rng.choice(obs["root_causes"])["id"],
                "fix": rng.choice(obs["fixes"])["id"]}

    return {
        "workgym/pair-programming-v0": pair_programming,
        "workgym/pr-review-v0": pr_review,
        "workgym/inbox-triage-v0": inbox_triage,
        "workgym/cron-ops-v0": cron_ops,
    }[env_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--env", default=None, help="single env id (default: all)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--base-seed", type=int, default=7)
    args = ap.parse_args()

    env_ids = [args.env] if args.env else workgym.list_envs()
    logger = EpisodeLogger(log_dir=args.log_dir)
    print("envs:", env_ids, "| seeds:", args.seeds, "| log:", logger.path)

    totals = {}
    for env_id in env_ids:
        policy = placeholder_policy(env_id)
        factories = [lambda e=env_id: workgym.make(e)] * args.seeds
        venv = VectorEnv(factories)
        try:
            rngs = [random.Random(args.base_seed * 1000 + i)
                    for i in range(args.seeds)]
            seeds = list(range(args.base_seed, args.base_seed + args.seeds))
            results = venv.reset(seeds=seeds)
            for (obs, info), seed in zip(results, seeds):
                logger.begin(env_id, info["episode_id"], seed=seed)
            actions = [policy(obs, rng) for (obs, _info), rng in zip(results, rngs)]
            step_outs = venv.step(actions)
            for (obs, info), (o2, r, term, trunc, i2), action in zip(
                    results, step_outs, actions):
                logger.step(info["episode_id"], action, r, term, trunc)
                logger.end(info["episode_id"], info=i2)
                totals.setdefault(env_id, []).append(r)
        finally:
            venv.close()

    for env_id, rs in totals.items():
        print(f"{env_id}: mean_reward={sum(rs)/len(rs):.3f} over {len(rs)} episodes")
    print("done ->", logger.path)


if __name__ == "__main__":
    main()
