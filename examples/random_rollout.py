"""Random-policy rollout demo: exercises the API and the episode logger.

Run: python3 examples/random_rollout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workgym
from workgym.log import EpisodeLogger


def random_policy_pair_programming(env, obs):
    # write a nonsense file, run tests once, then submit whatever is there
    env.step({"op": "write_file", "path": "solution.py",
              "content": "# random policy did not write a solution\n"})
    env.step({"op": "run_tests"})
    return {"op": "submit"}


def random_policy_pr_review(env, obs):
    return {"op": "submit", "findings": []}


def random_policy_inbox(env, obs):
    return {"op": "submit",
            "labels": {e["id"]: "NOISE" for e in obs["emails"]}}  # lazy baseline


POLICIES = {
    "workgym/pair-programming-v0": random_policy_pair_programming,
    "workgym/pr-review-v0": random_policy_pr_review,
    "workgym/inbox-triage-v0": random_policy_inbox,
}


def main():
    logger = EpisodeLogger()
    print("logging to:", logger.path)
    for env_id, policy in POLICIES.items():
        env = workgym.make(env_id)
        obs, info = env.reset(seed=42)
        ep = info["episode_id"]
        logger.begin(env_id, ep, seed=42)
        done = False
        total = 0.0
        while not done:
            action = policy(env, obs)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total += reward
            logger.step(ep, action, reward, term, trunc)
        logger.end(ep, info=info)
        env.close()
        print(f"{env_id}: total_reward={total:.2f} info={info}")
    print("done.")


if __name__ == "__main__":
    main()
