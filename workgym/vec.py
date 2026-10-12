"""Vectorized environments: run N envs in parallel via threads.

Stdlib only (concurrent.futures). Each env lives on its own thread;
envs must be thread-safe for concurrent use (each instance is used by
exactly one worker thread).
"""

from concurrent.futures import ThreadPoolExecutor


class VectorEnv:
    """Batched stepping over a list of env factories."""

    def __init__(self, env_fns, max_workers=None):
        if not env_fns:
            raise ValueError("need at least one env factory")
        self.env_fns = list(env_fns)
        self.num_envs = len(self.env_fns)
        self._pool = ThreadPoolExecutor(max_workers=max_workers or self.num_envs)
        self._envs = None

    def _ensure(self):
        if self._envs is None:
            self._envs = list(self._pool.map(lambda fn: fn(), self.env_fns))

    def reset(self, seeds=None):
        """Returns list of (obs, info), one per env."""
        self._ensure()
        seeds = seeds or [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("seeds length must match num_envs")
        return list(
            self._pool.map(lambda args: args[0].reset(seed=args[1]),
                           zip(self._envs, seeds))
        )

    def step(self, actions):
        """actions: list, one per env. Returns list of
        (obs, reward, terminated, truncated, info)."""
        self._ensure()
        if len(actions) != self.num_envs:
            raise ValueError("actions length must match num_envs")
        return list(
            self._pool.map(lambda args: args[0].step(args[1]),
                           zip(self._envs, actions))
        )

    def close(self):
        if self._envs is not None:
            list(self._pool.map(lambda e: e.close(), self._envs))
            self._envs = None
        self._pool.shutdown(wait=True)
