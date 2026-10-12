"""workgym/pair-programming-v0 — real coding tasks in a sandbox, verifiable reward.

Episode:
  reset  -> picks a task, writes starter files to a fresh temp dir.
            obs = {"task_id", "brief", "files": {path: content}, "step", "max_steps"}
            (hidden tests are never in the observation)
  step   -> action is a dict:
              {"op": "write_file", "path": "solution.py", "content": "..."}
              {"op": "run_tests"}   -> runs hidden tests, reports output, reward 0
              {"op": "submit"}      -> runs hidden tests, reward = pass rate, done
            Paths are jailed to the sandbox. Test runs are subprocess-isolated
            with a timeout; an infinite loop scores 0.

Reward is the fraction of hidden tests passing, in [0, 1]. The only way to
score is to write code that actually works.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

from ..core import Env, Text, Dict

_RUNNER_SRC = '''
import sys, traceback, importlib.util

spec = importlib.util.spec_from_file_location("test_hidden", "test_hidden.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["test_hidden"] = mod
try:
    spec.loader.exec_module(mod)
except Exception:
    print("WORKGYM_RESULT passed=0 failed=0 total=0")
    print("WORKGYM_FAIL <collection>")
    traceback.print_exc(limit=5)
    sys.exit(0)

names = sorted(n for n in dir(mod) if n.startswith("test_"))
passed = 0
failed = []
for n in names:
    try:
        getattr(mod, n)()
        passed += 1
    except Exception:
        failed.append(n)
        print("WORKGYM_FAIL " + n)
        traceback.print_exc(limit=4)

print("WORKGYM_RESULT passed=%d failed=%d total=%d" % (passed, len(failed), len(names)))
'''

_TASKS = [
    {
        "id": "retry",
        "brief": (
            "Implement `retry(fn, attempts=3)` in solution.py.\n"
            "- Call fn(). If it raises, retry until `attempts` total calls are used.\n"
            "- Return fn()'s return value on success.\n"
            "- Re-raise the last exception when attempts are exhausted.\n"
            "- attempts < 1 must raise ValueError."
        ),
        "starter": {"solution.py": 'def retry(fn, attempts=3):\n    """Call fn, retrying on exception up to `attempts` total calls."""\n    raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import retry

def test_success_first_try():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    assert retry(fn) == "ok"
    assert len(calls) == 1

def test_retries_then_succeeds():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("flaky")
        return 42
    assert retry(fn, attempts=3) == 42
    assert len(calls) == 3

def test_exhausts_and_reraises():
    n = []
    def fn():
        n.append(1)
        raise ValueError("always")
    try:
        retry(fn, attempts=2)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised ValueError")
    assert len(n) == 2

def test_attempts_one():
    n = []
    def fn():
        n.append(1)
        raise RuntimeError("x")
    try:
        retry(fn, attempts=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("should have raised")
    assert len(n) == 1

def test_bad_attempts():
    try:
        retry(lambda: 1, attempts=0)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised ValueError")

def test_returns_value_not_tuple():
    assert retry(lambda: [1, 2]) == [1, 2]
'''},
        "reference": {"solution.py": '''def retry(fn, attempts=3):
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
    raise last
'''},
    },
    {
        "id": "lru_cache",
        "brief": (
            "Implement `LRUCache` in solution.py.\n"
            "- `__init__(self, capacity)` with capacity >= 1.\n"
            "- `get(key)` returns the value, or -1 if absent.\n"
            "- `put(key, value)` inserts or updates.\n"
            "- When capacity is exceeded, evict the least recently used key.\n"
            "- Both get and put count as 'use' for recency."
        ),
        "starter": {"solution.py": 'class LRUCache:\n    def __init__(self, capacity):\n        raise NotImplementedError\n\n    def get(self, key):\n        raise NotImplementedError\n\n    def put(self, key, value):\n        raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import LRUCache

def test_basic():
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)          # evicts key 2
    assert c.get(2) == -1
    c.put(4, 4)          # evicts key 1
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4

def test_get_refreshes_recency():
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1   # 1 now most recent
    c.put(3, 3)            # evicts 2
    assert c.get(2) == -1
    assert c.get(1) == 1

def test_update_existing():
    c = LRUCache(2)
    c.put(1, 1); c.put(1, 10)
    assert c.get(1) == 10
    c.put(2, 2); c.put(3, 3)  # evicts 1
    assert c.get(1) == -1
    assert c.get(2) == 2

def test_capacity_one():
    c = LRUCache(1)
    c.put(1, 1)
    assert c.get(1) == 1
    c.put(2, 2)
    assert c.get(1) == -1
    assert c.get(2) == 2

def test_missing_key():
    c = LRUCache(3)
    assert c.get(99) == -1
'''},
        "reference": {"solution.py": '''from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return -1
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self._cap:
            self._d.popitem(last=False)
'''},
    },
    {
        "id": "deep_merge",
        "brief": (
            "Implement `deep_merge(a, b, concat_lists=False)` in solution.py.\n"
            "- Return a NEW dict merging b into a (never mutate the inputs).\n"
            "- Nested dicts merge recursively; b's scalar values win on conflict.\n"
            "- Lists are replaced by b's list, unless concat_lists=True (then a's + b's)."
        ),
        "starter": {"solution.py": 'def deep_merge(a, b, concat_lists=False):\n    """Recursively merge b into a, returning a new dict."""\n    raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import deep_merge
import copy

def test_flat():
    assert deep_merge({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}

def test_conflict_b_wins():
    assert deep_merge({"x": 1}, {"x": 2}) == {"x": 2}

def test_nested():
    a = {"db": {"host": "h", "port": 1}}
    b = {"db": {"port": 2, "tls": True}}
    assert deep_merge(a, b) == {"db": {"host": "h", "port": 2, "tls": True}}

def test_list_replace_default():
    assert deep_merge({"l": [1, 2]}, {"l": [3]}) == {"l": [3]}

def test_list_concat():
    assert deep_merge({"l": [1, 2]}, {"l": [3]}, concat_lists=True) == {"l": [1, 2, 3]}

def test_no_mutation():
    a = {"n": {"v": [1]}}
    b = {"n": {"v": [2], "w": 1}}
    ac, bc = copy.deepcopy(a), copy.deepcopy(b)
    deep_merge(a, b, concat_lists=True)
    assert a == ac and b == bc

def test_new_object():
    a = {"x": 1}
    out = deep_merge(a, {"y": 2})
    assert out == {"x": 1, "y": 2}
    assert "y" not in a
'''},
        "reference": {"solution.py": '''import copy

def deep_merge(a, b, concat_lists=False):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v, concat_lists=concat_lists)
        elif concat_lists and k in out and isinstance(out[k], list) and isinstance(v, list):
            out[k] = copy.deepcopy(out[k]) + copy.deepcopy(v)
        else:
            out[k] = copy.deepcopy(v)
    return out
'''},
    },
    {
        "id": "parse_duration",
        "brief": (
            "Implement `parse_duration(s)` in solution.py, returning int seconds.\n"
            '- Accepts combos like "1h30m", "45s", "2d", "1h 15m 30s".\n'
            '- A bare number like "90" means seconds.\n'
            "- Units: d (86400), h (3600), m (60), s (1).\n"
            "- Empty, whitespace-only, or malformed input raises ValueError."
        ),
        "starter": {"solution.py": 'def parse_duration(s):\n    """Parse a duration string into integer seconds."""\n    raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import parse_duration

def _bad(v):
    try:
        parse_duration(v)
    except ValueError:
        return
    raise AssertionError("expected ValueError for %r" % (v,))

def test_combos():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("1h 15m 30s") == 4530
    assert parse_duration("2d") == 172800

def test_single_units():
    assert parse_duration("45s") == 45
    assert parse_duration("1m") == 60
    assert parse_duration("3h") == 10800

def test_bare_number():
    assert parse_duration("90") == 90
    assert parse_duration("0") == 0

def test_invalid():
    _bad("")
    _bad("   ")
    _bad("abc")
    _bad("1x")
    _bad("h")
    _bad("--5s")
'''},
        "reference": {"solution.py": '''import re

_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}

def parse_duration(s):
    if not isinstance(s, str) or not s.strip():
        raise ValueError("empty duration")
    s = s.strip()
    if re.fullmatch(r"\\d+", s):
        return int(s)
    total, seen = 0, 0
    for m in re.finditer(r"(\\d+)\\s*([dhms])", s):
        total += int(m.group(1)) * _UNITS[m.group(2)]
        seen += 1
    rest = re.sub(r"(\\d+)\\s*([dhms])", "", s).strip()
    if seen == 0 or rest or re.search(r"-", s):
        raise ValueError("invalid duration: %r" % (s,))
    return total
'''},
    },
    {
        "id": "top_k",
        "brief": (
            "Implement `TopK` in solution.py.\n"
            "- `__init__(self, k)` with k >= 1.\n"
            "- `add(x)` records a value.\n"
            "- `top()` returns the k largest values seen so far, descending.\n"
            "  (Fewer than k seen -> return all of them, descending.)"
        ),
        "starter": {"solution.py": 'class TopK:\n    def __init__(self, k):\n        raise NotImplementedError\n\n    def add(self, x):\n        raise NotImplementedError\n\n    def top(self):\n        raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import TopK

def test_basic():
    t = TopK(3)
    for x in [5, 1, 9, 3, 7]:
        t.add(x)
    assert t.top() == [9, 7, 5]

def test_fewer_than_k():
    t = TopK(5)
    t.add(2); t.add(8)
    assert t.top() == [8, 2]

def test_duplicates():
    t = TopK(2)
    for x in [4, 4, 4]:
        t.add(x)
    assert t.top() == [4, 4]

def test_negatives():
    t = TopK(2)
    for x in [-1, -9, -3]:
        t.add(x)
    assert t.top() == [-1, -3]

def test_k_one():
    t = TopK(1)
    for x in [2, 9, 4]:
        t.add(x)
    assert t.top() == [9]

def test_empty():
    assert TopK(3).top() == []
'''},
        "reference": {"solution.py": '''import heapq

class TopK:
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self._k = k
        self._h = []

    def add(self, x):
        if len(self._h) < self._k:
            heapq.heappush(self._h, x)
        elif x > self._h[0]:
            heapq.heapreplace(self._h, x)

    def top(self):
        return sorted(self._h, reverse=True)
'''},
    },
    {
        "id": "sliding_median",
        "brief": (
            "Implement `SlidingMedian` in solution.py.\n"
            "- `__init__(self, window)` with window >= 1.\n"
            "- `add(x)` pushes a value, keeping only the last `window` values.\n"
            "- `median()` returns the median of the current window as a float.\n"
            "  (Even count -> average of the two middle values.)\n"
            "- Calling median() with no values raises ValueError."
        ),
        "starter": {"solution.py": 'class SlidingMedian:\n    def __init__(self, window):\n        raise NotImplementedError\n\n    def add(self, x):\n        raise NotImplementedError\n\n    def median(self):\n        raise NotImplementedError\n'},
        "tests": {"test_hidden.py": '''
from solution import SlidingMedian

def _bad_empty():
    m = SlidingMedian(3)
    try:
        m.median()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

def test_odd_window():
    m = SlidingMedian(3)
    for x in [5, 1, 9]:
        m.add(x)
    assert m.median() == 5.0

def test_even_window():
    m = SlidingMedian(4)
    for x in [1, 2, 3, 4]:
        m.add(x)
    assert m.median() == 2.5

def test_slides():
    m = SlidingMedian(3)
    for x in [1, 2, 3, 100]:
        m.add(x)
    assert m.median() == 3.0  # window is [2, 3, 100]

def test_window_one():
    m = SlidingMedian(1)
    m.add(7); assert m.median() == 7.0
    m.add(2); assert m.median() == 2.0

def test_empty_raises():
    _bad_empty()
'''},
        "reference": {"solution.py": '''from collections import deque

class SlidingMedian:
    def __init__(self, window):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._w = window
        self._d = deque()

    def add(self, x):
        self._d.append(x)
        while len(self._d) > self._w:
            self._d.popleft()

    def median(self):
        if not self._d:
            raise ValueError("no values")
        s = sorted(self._d)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0
'''},
    },
]


class PairProgrammingEnv(Env):
    """workgym/pair-programming-v0."""

    env_id = "workgym/pair-programming-v0"

    def __init__(self, task_ids=None, max_steps=20, test_timeout=30):
        super().__init__()
        self.task_ids = list(task_ids) if task_ids else [t["id"] for t in _TASKS]
        unknown = set(self.task_ids) - {t["id"] for t in _TASKS}
        if unknown:
            raise ValueError(f"unknown task ids: {sorted(unknown)}")
        self.max_steps = max_steps
        self.test_timeout = test_timeout
        self.action_space = Text(max_length=1_000_000)
        self.observation_space = Dict({
            "task_id": Text(), "brief": Text(), "files": Text(),
            "step": Text(), "max_steps": Text(), "last_test_output": Text(),
        })
        self._sandbox = None
        self._task = None

    # -- internals ---------------------------------------------------------
    def _task_by_id(self, tid):
        for t in _TASKS:
            if t["id"] == tid:
                return t
        raise KeyError(tid)

    def _write(self, relpath, content):
        # jail paths inside the sandbox
        dest = os.path.normpath(os.path.join(self._sandbox, relpath))
        if not dest.startswith(os.path.realpath(self._sandbox) + os.sep):
            raise ValueError(f"path escapes sandbox: {relpath}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)

    def _read_all(self):
        out = {}
        for root, _dirs, files in os.walk(self._sandbox):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, self._sandbox)
                if rel in ("_runner.py", "test_hidden.py"):
                    continue
                try:
                    with open(full, encoding="utf-8") as f:
                        out[rel] = f.read()
                except (OSError, UnicodeDecodeError):
                    out[rel] = "<unreadable>"
        return out

    def _obs(self, last_test_output=""):
        return {
            "task_id": self._task["id"],
            "brief": self._task["brief"],
            "files": self._read_all(),
            "step": self._step_count,
            "max_steps": self.max_steps,
            "last_test_output": last_test_output[-4000:],
        }

    def _run_tests(self):
        """Run hidden tests in a subprocess. Returns (passed, failed, total, output)."""
        try:
            proc = subprocess.run(
                [sys.executable, "_runner.py"],
                cwd=self._sandbox,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            return 0, 0, 0, "TIMEOUT: test run exceeded %ds" % self.test_timeout
        except OSError as e:
            return 0, 0, 0, "RUNNER ERROR: %s" % e
        m = re.search(r"WORKGYM_RESULT passed=(\d+) failed=(\d+) total=(\d+)", out)
        if not m:
            return 0, 0, 0, "RUNNER ERROR: no result line\n" + out[-2000:]
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), out

    # -- Env API ------------------------------------------------------------
    def _reset(self, seed):
        self.close()
        tid = self._rng.choice(self.task_ids)
        self._task = self._task_by_id(tid)
        self._sandbox = tempfile.mkdtemp(prefix="workgym-pair-")
        with open(os.path.join(self._sandbox, "_runner.py"), "w") as f:
            f.write(_RUNNER_SRC)
        with open(os.path.join(self._sandbox, "test_hidden.py"), "w") as f:
            f.write(self._task["tests"]["test_hidden.py"])
        for path, content in self._task["starter"].items():
            self._write(path, content)
        return self._obs(), {"task_id": tid}

    def _step(self, action):
        if not isinstance(action, dict) or "op" not in action:
            return self._obs(), 0.0, False, False, {
                "error": "action must be a dict with an 'op' key"}
        op = action["op"]
        if op == "write_file":
            try:
                self._write(action["path"], action.get("content", ""))
            except (ValueError, OSError, KeyError) as e:
                return self._obs(), 0.0, False, False, {"error": str(e)}
            truncated = self._step_count >= self.max_steps
            return self._obs(), 0.0, False, truncated, {}
        if op == "run_tests":
            passed, failed, total, out = self._run_tests()
            truncated = self._step_count >= self.max_steps
            return self._obs(out), 0.0, False, truncated, {
                "passed": passed, "failed": failed, "total": total}
        if op == "submit":
            passed, failed, total, out = self._run_tests()
            reward = (passed / total) if total else 0.0
            return self._obs(out), reward, True, False, {
                "task_id": self._task["id"],
                "passed": passed, "failed": failed, "total": total}
        return self._obs(), 0.0, False, False, {"error": f"unknown op {op!r}"}

    def close(self):
        if self._sandbox and os.path.isdir(self._sandbox):
            shutil.rmtree(self._sandbox, ignore_errors=True)
        self._sandbox = None


def _register():
    from .. import register
    register(PairProgrammingEnv.env_id, PairProgrammingEnv)


_register()
