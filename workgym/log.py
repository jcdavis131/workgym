"""Episode logging: append-only JSONL, one record per episode.

Schema is deliberately close to the Dottie trace pipeline so these logs can
feed grpo_collect with minimal translation:

    pair_sessions.jsonl -> dottie_telemetry.jsonl -> grpo_collect.py -> pref_pairs

Each record:
{
  "ts": iso8601,
  "env_id": "workgym/pair-programming-v0",
  "episode_id": "abc123",
  "seed": 7,
  "steps": [
    {"t": 0, "action": {...}, "reward": 0.0, "terminated": false, "truncated": false},
    ...
  ],
  "total_reward": 0.83,
  "terminated": true,
  "info": {...}          # final info dict (verifier scores, test counts, ...)
}

Actions/observations are NOT logged by default (they can be large and may
contain sensitive repo content). Pass log_io=True to include them.
"""

import datetime
import json
import os
import threading


def _default_log_dir():
    return os.environ.get(
        "WORKGYM_LOG_DIR",
        os.path.join(os.path.expanduser("~"), "workspace", "workgym", "episodes"),
    )


class EpisodeLogger:
    def __init__(self, log_dir=None, filename="workgym_episodes.jsonl"):
        self.log_dir = log_dir or _default_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(self.log_dir, filename)
        self._lock = threading.Lock()
        self._buf = []
        self._active = {}  # episode_id -> record under construction

    def begin(self, env_id, episode_id, seed=None):
        rec = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "env_id": env_id,
            "episode_id": episode_id,
            "seed": seed,
            "steps": [],
            "total_reward": 0.0,
            "terminated": False,
        }
        self._buf.append(rec)
        self._active[episode_id] = rec
        return rec

    def step(self, episode_id, action, reward, terminated, truncated,
             obs=None, log_io=False):
        rec = self._active.get(episode_id)
        if rec is None:
            raise KeyError(f"no active episode {episode_id}")
        entry = {
            "t": len(rec["steps"]),
            "action": action if log_io else _summarize(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if log_io and obs is not None:
            entry["obs"] = _json_safe(obs)
        rec["steps"].append(entry)
        rec["total_reward"] = round(rec["total_reward"] + float(reward), 6)
        rec["terminated"] = bool(terminated)
        return entry

    def end(self, episode_id, info=None):
        rec = self._active.pop(episode_id, None)
        if rec is None:
            raise KeyError(f"no active episode {episode_id}")
        if info:
            rec["info"] = _json_safe(info)
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec


def _summarize(action):
    if isinstance(action, dict):
        op = action.get("op", action.get("type", "?"))
        keys = sorted(action.keys())
        return {"op": op, "keys": keys}
    return {"type": type(action).__name__}


def _json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)[:2000]
