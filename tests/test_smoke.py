"""Smoke tests: the API contract holds, and gold solutions score 1.0.

Run: python3 -m unittest discover -s tests -v   (from the workgym root)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workgym
from workgym.envs.pair_programming import _TASKS
from workgym.envs.pr_review import _DIFFS
from workgym.envs.inbox_triage import _INBOXES
from workgym.envs.cron_ops import _SCENARIOS
from workgym.log import EpisodeLogger
from workgym.vec import VectorEnv
from workgym.train.grpo_adapter import convert


class TestRegistry(unittest.TestCase):
    def test_four_envs_registered(self):
        self.assertEqual(workgym.list_envs(), [
            "workgym/cron-ops-v0",
            "workgym/inbox-triage-v0",
            "workgym/pair-programming-v0",
            "workgym/pr-review-v0",
        ])

    def test_make_unknown_raises(self):
        with self.assertRaises(ValueError):
            workgym.make("workgym/does-not-exist-v0")


class TestPairProgramming(unittest.TestCase):
    def _submit_reference(self, task_id):
        env = workgym.make("workgym/pair-programming-v0", task_ids=[task_id])
        try:
            obs, info = env.reset(seed=1)
            self.assertEqual(info["task_id"], task_id)
            task = next(t for t in _TASKS if t["id"] == task_id)
            for path, content in task["reference"].items():
                o2, r, term, trunc, info2 = env.step(
                    {"op": "write_file", "path": path, "content": content})
                self.assertFalse(term)
            obs, reward, term, trunc, info = env.step({"op": "submit"})
            return reward, info
        finally:
            env.close()

    def test_reference_solutions_score_one(self):
        for task in _TASKS:
            reward, info = self._submit_reference(task["id"])
            self.assertEqual(reward, 1.0,
                             f"task {task['id']}: reward={reward}, info={info}")

    def test_starter_scores_zero(self):
        env = workgym.make("workgym/pair-programming-v0", task_ids=["retry"])
        try:
            env.reset(seed=1)
            _o, reward, term, _tr, info = env.step({"op": "submit"})
            self.assertTrue(term)
            self.assertLess(reward, 1.0)
        finally:
            env.close()

    def test_run_tests_reports_output(self):
        env = workgym.make("workgym/pair-programming-v0", task_ids=["top_k"])
        try:
            env.reset(seed=1)
            task = next(t for t in _TASKS if t["id"] == "top_k")
            for path, content in task["reference"].items():
                env.step({"op": "write_file", "path": path, "content": content})
            obs, r, term, trunc, info = env.step({"op": "run_tests"})
            self.assertFalse(term)
            self.assertEqual(r, 0.0)
            self.assertEqual(info["passed"], info["total"])
            self.assertIn("WORKGYM_RESULT", obs["last_test_output"])
        finally:
            env.close()

    def test_path_escape_blocked(self):
        env = workgym.make("workgym/pair-programming-v0", task_ids=["retry"])
        try:
            env.reset(seed=1)
            _o, _r, _t, _tr, info = env.step(
                {"op": "write_file", "path": "../../evil.py", "content": "x"})
            self.assertIn("error", info)
        finally:
            env.close()

    def test_max_steps_truncates(self):
        env = workgym.make("workgym/pair-programming-v0", task_ids=["retry"],
                           max_steps=2)
        try:
            env.reset(seed=1)
            env.step({"op": "write_file", "path": "a.py", "content": "x=1"})
            _o, _r, _t, trunc, _i = env.step({"op": "write_file", "path": "b.py", "content": "x=2"})
            self.assertTrue(trunc)
        finally:
            env.close()


class TestPrReview(unittest.TestCase):
    def test_gold_findings_score_one(self):
        for d in _DIFFS:
            env = workgym.make("workgym/pr-review-v0", diff_ids=[d["id"]])
            try:
                env.reset(seed=1)
                findings = [
                    {"file": b["file"], "line": b["line"],
                     "category": b["category"], "comment": b["note"]}
                    for b in d["bugs"]
                ]
                _o, reward, term, _tr, info = env.step(
                    {"op": "submit", "findings": findings})
                self.assertTrue(term)
                self.assertEqual(reward, 1.0,
                                 f"diff {d['id']}: reward={reward}, info={info}")
            finally:
                env.close()

    def test_empty_submit_on_buggy_diff_scores_zero(self):
        env = workgym.make("workgym/pr-review-v0", diff_ids=["sql-swallow"])
        try:
            env.reset(seed=1)
            _o, reward, _t, _tr, _i = env.step({"op": "submit", "findings": []})
            self.assertEqual(reward, 0.0)
        finally:
            env.close()

    def test_false_positive_hurts(self):
        env = workgym.make("workgym/pr-review-v0", diff_ids=["clean-rename"])
        try:
            env.reset(seed=1)
            _o, reward, _t, _tr, _i = env.step({"op": "submit", "findings": [
                {"file": "src/names.js", "line": 2, "category": "correctness",
                 "comment": "bogus"}]})
            self.assertEqual(reward, 0.0)
        finally:
            env.close()


class TestInboxTriage(unittest.TestCase):
    def test_gold_labels_score_one(self):
        for b in _INBOXES:
            env = workgym.make("workgym/inbox-triage-v0", inbox_ids=[b["id"]])
            try:
                env.reset(seed=1)
                labels = {e["id"]: e["gold"] for e in b["emails"]}
                _o, reward, term, _tr, info = env.step(
                    {"op": "submit", "labels": labels})
                self.assertTrue(term)
                self.assertEqual(reward, 1.0,
                                 f"inbox {b['id']}: reward={reward}, info={info}")
            finally:
                env.close()

    def test_all_noise_scores_partially(self):
        env = workgym.make("workgym/inbox-triage-v0", inbox_ids=["inbox-0"])
        try:
            obs, _i = env.reset(seed=1)
            labels = {e["id"]: "NOISE" for e in obs["emails"]}
            _o, reward, _t, _tr, info = env.step({"op": "submit", "labels": labels})
            self.assertGreater(reward, 0.0)
            self.assertLess(reward, 1.0)
            self.assertEqual(info["correct"] + (info["total"] - info["correct"]),
                             info["total"])
        finally:
            env.close()


class TestVecAndLog(unittest.TestCase):
    def test_vector_env(self):
        venv = VectorEnv(
            [lambda: workgym.make("workgym/pr-review-v0"),
             lambda: workgym.make("workgym/inbox-triage-v0")])
        try:
            outs = venv.reset(seeds=[1, 2])
            self.assertEqual(len(outs), 2)
            res = venv.step([
                {"op": "submit", "findings": []},
                {"op": "submit", "labels": {}},
            ])
            self.assertEqual(len(res), 2)
            for (_o, r, _te, _tr, _i) in res:
                self.assertGreaterEqual(r, 0.0)
                self.assertLessEqual(r, 1.0)
        finally:
            venv.close()

    def test_episode_logger(self):
        with tempfile.TemporaryDirectory() as td:
            lg = EpisodeLogger(log_dir=td)
            lg.begin("workgym/pr-review-v0", "ep1", seed=3)
            lg.step("ep1", {"op": "submit"}, 0.5, False, False)
            lg.step("ep1", {"op": "submit"}, 0.5, True, False)
            rec = lg.end("ep1", info={"num_bugs": 2})
            self.assertEqual(rec["total_reward"], 1.0)
            with open(os.path.join(td, "workgym_episodes.jsonl")) as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 1)
            back = json.loads(lines[0])
            self.assertEqual(back["episode_id"], "ep1")
            self.assertEqual(len(back["steps"]), 2)


class TestCronOps(unittest.TestCase):
    def test_gold_diagnoses_score_one(self):
        for s in _SCENARIOS:
            env = workgym.make("workgym/cron-ops-v0", scenario_ids=[s["id"]])
            try:
                env.reset(seed=1)
                _o, reward, term, _tr, info = env.step({"op": "submit",
                    "root_cause": s["gold"]["root_cause"],
                    "fix": s["gold"]["fix"]})
                self.assertTrue(term)
                self.assertEqual(reward, 1.0,
                                 f"scenario {s['id']}: reward={reward}, info={info}")
                self.assertTrue(info["root_cause_correct"])
                self.assertTrue(info["fix_correct"])
            finally:
                env.close()

    def test_half_credit(self):
        env = workgym.make("workgym/cron-ops-v0", scenario_ids=["dead-trigger"])
        try:
            env.reset(seed=1)
            _o, reward, _t, _tr, info = env.step({"op": "submit",
                "root_cause": "rc-a", "fix": "fx-z"})  # right cause, wrong fix
            self.assertEqual(reward, 0.5)
            self.assertTrue(info["root_cause_correct"])
            self.assertFalse(info["fix_correct"])
        finally:
            env.close()

    def test_wrong_scores_zero(self):
        env = workgym.make("workgym/cron-ops-v0", scenario_ids=["stale-feed"])
        try:
            env.reset(seed=1)
            _o, reward, _t, _tr, _i = env.step({"op": "submit",
                "root_cause": "rc-c", "fix": "fx-z"})
            self.assertEqual(reward, 0.0)
        finally:
            env.close()


class TestGrpoAdapter(unittest.TestCase):
    def test_convert_schema(self):
        with tempfile.TemporaryDirectory() as td:
            ep_path = os.path.join(td, "eps.jsonl")
            out_path = os.path.join(td, "tel.jsonl")
            rec = {"ts": "2026-10-12T00:00:00+00:00", "env_id": "workgym/pair-programming-v0",
                   "episode_id": "abc123", "seed": 7,
                   "steps": [{"t": 0, "action": {"op": "submit"}, "reward": 1.0,
                              "terminated": True, "truncated": False}],
                   "total_reward": 1.0, "terminated": True,
                   "info": {"task_id": "retry", "passed": 6, "total": 6}}
            with open(ep_path, "w") as f:
                f.write(json.dumps(rec) + "\n")
            n_in, n_out = convert(ep_path, out_path)
            self.assertEqual((n_in, n_out), (1, 1))
            with open(out_path) as f:
                tel = json.loads(f.read().strip())
            for key in ("prompt", "prompt_id", "completion", "trace_id",
                        "rl_return", "verdict", "source"):
                self.assertIn(key, tel)
            self.assertEqual(tel["rl_return"], 1.0)
            self.assertEqual(tel["verdict"], "pass")
            self.assertIn("retry", tel["prompt"])
            # same task -> same prompt_id (GRPO grouping key)
            self.assertEqual(tel["prompt_id"], __import__("hashlib").sha1(
                tel["prompt"].encode()).hexdigest()[:16])


if __name__ == "__main__":
    unittest.main()
