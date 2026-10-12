"""WorkGym core: a minimal Gymnasium-compatible RL interface for work workflows.

Environments are real work tasks (coding, review, triage), not games.
Observations are structured dicts (brief + state + tool outputs), actions are
structured dicts or text, and rewards are always verifiable — never vibes.

Stdlib only. No third-party dependencies.
"""

import abc
import random


# ---------------------------------------------------------------------------
# Spaces (lightweight descriptors; validated, not sampled from, in v1)
# ---------------------------------------------------------------------------
class Space:
    """Base space. `contains(x)` validates a value against the space."""

    def contains(self, x):
        raise NotImplementedError


class Text(Space):
    def __init__(self, max_length=65536):
        self.max_length = max_length

    def contains(self, x):
        return isinstance(x, str) and len(x) <= self.max_length

    def __repr__(self):
        return f"Text(max_length={self.max_length})"


class Discrete(Space):
    def __init__(self, n):
        if n <= 0:
            raise ValueError("Discrete n must be positive")
        self.n = n

    def contains(self, x):
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"


class Dict(Space):
    def __init__(self, spaces):
        self.spaces = dict(spaces)

    def contains(self, x):
        return (
            isinstance(x, dict)
            and set(x.keys()) == set(self.spaces.keys())
            and all(sp.contains(x[k]) for k, sp in self.spaces.items())
        )

    def __repr__(self):
        return f"Dict({self.spaces!r})"


# ---------------------------------------------------------------------------
# Env base class (Gymnasium-style API)
# ---------------------------------------------------------------------------
class Env(abc.ABC):
    """Base environment.

    reset(seed) -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)

    reward is always a float in [0, 1].
    info always carries at least {"env_id": ..., "episode_id": ...}.
    """

    action_space: Space
    observation_space: Space
    env_id: str = "workgym/unnamed-v0"

    def __init__(self):
        self._rng = random.Random()
        self._episode_id = None
        self._step_count = 0

    # -- to implement ------------------------------------------------------
    @abc.abstractmethod
    def _reset(self, seed):
        """Return (obs, info). Called with self._rng freshly seeded."""

    @abc.abstractmethod
    def _step(self, action):
        """Return (obs, reward, terminated, truncated, info)."""

    # -- public API ---------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()
        import uuid as _uuid

        self._episode_id = _uuid.uuid4().hex[:12]
        self._step_count = 0
        obs, info = self._reset(seed)
        info = dict(info or {})
        info.setdefault("env_id", self.env_id)
        info.setdefault("episode_id", self._episode_id)
        return obs, info

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = self._step(action)
        reward = max(0.0, min(1.0, float(reward)))
        info = dict(info or {})
        info.setdefault("env_id", self.env_id)
        info.setdefault("episode_id", self._episode_id)
        info.setdefault("step", self._step_count)
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # -- helpers ------------------------------------------------------------
    def seed(self, seed):
        self._rng = random.Random(seed)
