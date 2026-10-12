"""WorkGym registry: register environments, build them by id."""

from .core import Env

_REGISTRY = {}


def register(env_id, entry_point):
    """Register an env id. entry_point is a callable returning an Env instance."""
    if env_id in _REGISTRY:
        raise ValueError(f"env id already registered: {env_id}")
    _REGISTRY[env_id] = entry_point


def make(env_id, **kwargs):
    """Build an environment by id, e.g. workgym.make("workgym/pair-programming-v0")."""
    try:
        factory = _REGISTRY[env_id]
    except KeyError:
        raise ValueError(
            f"unknown env id {env_id!r}; registered: {sorted(_REGISTRY)}"
        ) from None
    env = factory(**kwargs)
    if not isinstance(env, Env):
        raise TypeError(f"factory for {env_id!r} did not return an Env")
    return env


def list_envs():
    """Sorted list of registered env ids."""
    return sorted(_REGISTRY)


# Built-in environments register themselves on import.
from .envs import pair_programming  # noqa: E402,F401
from .envs import pr_review  # noqa: E402,F401
from .envs import inbox_triage  # noqa: E402,F401
from .envs import cron_ops  # noqa: E402,F401

__version__ = "0.1.0"
