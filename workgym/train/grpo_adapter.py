"""WorkGym -> GRPO adapter.

Converts WorkGym episode logs (workgym/log.py JSONL) into the telemetry
schema consumed by dottie/apps/ava-factory/dottie/pipeline/grpo_collect.py:

    prompt, completion, rl_return, trace_id, verdict, logp_new, logp_old,
    entropy, source

GRPO grouping works the way grpo_collect expects: episodes that share a
prompt (same env + same task id) land in one group via prompt_id hashing,
and group_advantages() is computed within the group. So: roll out each task
K times (different seeds — VectorEnv makes this cheap), then run this
adapter, then grpo_collect.

Usage:
    python3 -m workgym.train.grpo_adapter \\
        --episodes episodes/workgym_episodes.jsonl \\
        --out reports/workgym_telemetry.jsonl

    # then, from the dottie ava-factory dir:
    python3 dottie/pipeline/grpo_collect.py \\
        --in /path/to/workgym/reports --out runs/grpo_pref
    # (grpo_collect reads dottie_telemetry.jsonl or ava_telemetry.jsonl from
    #  --in, so name the adapter output accordingly or merge it in.)

Stdlib only.
"""

import argparse
import hashlib
import json
import os
import sys


def sha16(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _task_key(rec):
    info = rec.get("info") or {}
    for k in ("task_id", "diff_id", "inbox_id", "scenario_id"):
        if info.get(k):
            return str(info[k])
    return rec.get("episode_id", "unknown")


def _completion_summary(rec):
    ops = []
    for st in rec.get("steps", []):
        a = st.get("action", {})
        if isinstance(a, dict):
            ops.append(str(a.get("op", "?")))
        else:
            ops.append(str(a)[:40])
    info = rec.get("info") or {}
    extras = []
    for k in ("passed", "total", "tp", "fp", "fn", "correct",
              "root_cause_correct", "fix_correct"):
        if k in info:
            extras.append(f"{k}={info[k]}")
    tail = (" | " + " ".join(extras)) if extras else ""
    return " -> ".join(ops) + f" | reward={rec.get('total_reward', 0.0):.2f}" + tail


def convert(episodes_path, out_path, pass_threshold=0.8):
    n_in, n_out = 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(episodes_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            env_id = rec.get("env_id", "workgym/unknown")
            prompt = f"{env_id}::{_task_key(rec)}"
            total = float(rec.get("total_reward", 0.0))
            rollout = {
                "prompt": prompt,
                "prompt_id": sha16(prompt),
                "completion": _completion_summary(rec),
                "trace_id": rec.get("episode_id") or sha16(prompt + str(n_in)),
                "rl_return": total,
                "logp_new": [],
                "logp_old": [],
                "entropy": 0.0,
                "verdict": "pass" if total >= pass_threshold else "fail",
                "source": "workgym",
                "seed": rec.get("seed"),
            }
            fout.write(json.dumps(rollout) + "\n")
            n_out += 1
    return n_in, n_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", required=True,
                    help="workgym episode JSONL (from EpisodeLogger)")
    ap.add_argument("--out", required=True,
                    help="output telemetry JSONL for grpo_collect")
    ap.add_argument("--pass-threshold", type=float, default=0.8,
                    help="total_reward >= this counts as pass (default 0.8)")
    args = ap.parse_args(argv)
    n_in, n_out = convert(args.episodes, args.out, args.pass_threshold)
    print(f"converted {n_out}/{n_in} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
