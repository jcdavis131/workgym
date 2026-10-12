"""workgym/cron-ops-v0 — diagnose a failing scheduled job from its logs.

Each episode presents a real-style ops incident: a cron job or hook that is
failing, silent, or misbehaving, with its recent logs and config. The agent
picks the root cause and the fix from the listed candidates.

Reward: 0.5 for the correct root cause + 0.5 for the correct fix.
Single-step: one submit ends the episode.

Scenarios are modeled on real incidents from this fleet (dead trigger files,
stale feeds, port conflicts, RAM pressure, silenced cron output) with the
identifying details fictionalized.
"""

from ..core import Env, Text, Dict

_SCENARIOS = [
    {
        "id": "dead-trigger",
        "title": "triage-watch hook: 0 events in 72h",
        "context": (
            "Hook `triage_watch` fires when /tmp/triage_trigger.json changes. "
            "It pages the on-call agent on new inbox items."
        ),
        "logs": (
            "[2026-10-09 00:00:01] triage_watch: poll #1, mtime unchanged\n"
            "[2026-10-09 12:00:01] triage_watch: poll #720, mtime unchanged\n"
            "[2026-10-10 00:00:01] triage_watch: poll #1441, mtime unchanged\n"
            "[2026-10-10 12:00:01] triage_watch: poll #2160, mtime unchanged, 0 events total\n"
            "$ ls -la /tmp/triage_trigger.json\n"
            "-rw-r--r-- 1 hatch hatch 2 Oct  7 09:12 /tmp/triage_trigger.json\n"
            "$ grep -r \"triage_trigger\" ~/workspace --include=\"*.py\" --include=\"*.sh\" -l\n"
            "(no output — nothing references the file)"
        ),
        "config": "hook: {name: triage_watch, path: /tmp/triage_trigger.json, interval: 60s}",
        "root_causes": [
            {"id": "rc-a", "text": "No producer ever writes the trigger file — the hook watches a file nothing creates/updates"},
            {"id": "rc-b", "text": "The 60s poll interval is too long; events arrive faster than polls"},
            {"id": "rc-c", "text": "File permissions prevent the hook from reading the trigger"},
        ],
        "fixes": [
            {"id": "fx-x", "text": "Disable the hook — it is a dead stub providing no signal"},
            {"id": "fx-y", "text": "Wire a real producer: have the inbox poller touch the trigger file on new mail"},
            {"id": "fx-z", "text": "chmod 666 the trigger file so the hook can read it"},
        ],
        "gold": {"root_cause": "rc-a", "fix": "fx-x"},
    },
    {
        "id": "stale-feed",
        "title": "brief-poller: 960 polls/day, feed unchanged since Aug 18",
        "context": (
            "Cron `brief-poller` fetches an external news feed every 90 seconds "
            "and rebuilds the morning brief when the feed changes."
        ),
        "logs": (
            "[2026-10-10 08:00:02] brief-poller: fetched 200 OK, etag unchanged (214th day)\n"
            "[2026-10-10 08:01:32] brief-poller: fetched 200 OK, etag unchanged\n"
            "[2026-10-10 08:03:02] brief-poller: fetched 200 OK, etag unchanged\n"
            "$ curl -sI https://feeds.example.com/morning | head -3\n"
            "HTTP/2 200\n"
            "last-modified: Tue, 18 Aug 2026 07:02:11 GMT\n"
            "etag: \"stale-feed-v1\"\n"
            "$ tail -2 briefs.log\n"
            "[2026-08-18] brief rebuilt (feed changed)\n"
            "[2026-08-19] no change — skipping (day 1 of 54)"
        ),
        "config": "cron: */1.5 * * * * /opt/briefs/poll.sh   # every 90 seconds",
        "root_causes": [
            {"id": "rc-a", "text": "The upstream feed has been dead since Aug 18 — polling a corpse 960x/day"},
            {"id": "rc-b", "text": "The 90-second interval is misconfigured; it should be hourly"},
            {"id": "rc-c", "text": "Auth credentials for the feed expired, so updates are blocked"},
        ],
        "fixes": [
            {"id": "fx-x", "text": "Disable the poller until the feed is live again — stop burning 960 polls/day on nothing"},
            {"id": "fx-y", "text": "Change the interval to hourly to reduce waste"},
            {"id": "fx-z", "text": "Rotate the feed credentials and retry"},
        ],
        "gold": {"root_cause": "rc-a", "fix": "fx-x"},
    },
    {
        "id": "port-conflict",
        "title": "preview-server: 'Address already in use' on every start",
        "context": (
            "Cron `preview-server` starts a local HTTP server on port 8932 for the "
            "design-review page. It has failed 6 starts in a row."
        ),
        "logs": (
            "[2026-10-12 13:20:01] preview-server: starting on :8932\n"
            "Traceback (most recent call last):\n"
            "  File \"http/server.py\", line 136, in server_bind\n"
            "OSError: [Errno 98] Address already in use\n"
            "[2026-10-12 13:25:01] preview-server: starting on :8932\n"
            "OSError: [Errno 98] Address already in use\n"
            "$ ss -tlnp | grep 8932\n"
            "LISTEN 0  5  0.0.0.0:8932  users:((\"python3\",pid=118811,fd=3))\n"
            "$ ps -o etime= -p 118811\n"
            "  3-02:14:55   # started 3 days ago"
        ),
        "config": "cron: */5 * * * * /opt/preview/serve.sh  # blind restart every 5 min",
        "root_causes": [
            {"id": "rc-a", "text": "A previous server instance (pid 118811, 3 days old) still holds the port"},
            {"id": "rc-b", "text": "Port 8932 is hardcoded; it should be configurable"},
            {"id": "rc-c", "text": "The firewall is blocking loopback connections to the port"},
        ],
        "fixes": [
            {"id": "fx-x", "text": "Kill the stale process, then start the server once"},
            {"id": "fx-y", "text": "Move the server to a different port"},
            {"id": "fx-z", "text": "Enable SO_REUSEPORT so two servers can share the port"},
        ],
        "gold": {"root_cause": "rc-a", "fix": "fx-x"},
    },
    {
        "id": "ram-pressure",
        "title": "gpu-trainer: heartbeat flags blocked_ram, free RAM declining",
        "context": (
            "The GPU training box runs jobs from a queue. The watchdog reports "
            "free RAM falling steadily and has started refusing new jobs."
        ),
        "logs": (
            "[14:02] watchdog: free RAM 1253 MB — ok\n"
            "[14:31] watchdog: free RAM 1044 MB — ok\n"
            "[15:04] watchdog: free RAM  913 MB — ok\n"
            "[15:34] watchdog: free RAM  824 MB — BLOCKED_RAM, refusing new jobs\n"
            "$ queue status\n"
            "  running: train-unified-g2 (8.1 GB VRAM), train-hoops-v9 (7.4 GB VRAM),\n"
            "           backfill-gamelogs, embed-rebuild, eval-challenger\n"
            "  note: box has 16 GB system RAM; each data job holds ~1.5 GB RSS"
        ),
        "config": "runner: {max_concurrent: 5, ram_gate_mb: 900}",
        "root_causes": [
            {"id": "rc-a", "text": "A memory leak in the runner — RAM falls with no new jobs"},
            {"id": "rc-b", "text": "Over-scheduled concurrency: 2 training + 3 data jobs exceed comfortable RAM headroom"},
            {"id": "rc-c", "text": "Normal growth — the box just needs more RAM installed"},
        ],
        "fixes": [
            {"id": "fx-x", "text": "Drain the queue to 1 concurrent job and re-check; add a RAM gate before dispatching heavy training"},
            {"id": "fx-y", "text": "Restart the runner to clear the leak"},
            {"id": "fx-z", "text": "Order more RAM for the box"},
        ],
        "gold": {"root_cause": "rc-b", "fix": "fx-x"},
    },
    {
        "id": "silent-cron",
        "title": "nightly-report: exit 0 every night, zero output for 2 weeks",
        "context": (
            "Cron `nightly-report` emails a summary every night. It exits 0, but no "
            "email has gone out in 14 days and the log file is empty."
        ),
        "logs": (
            "$ grep nightly-report /var/log/cron | tail -3\n"
            "Oct 10 02:00:01 CRON: (hatch) CMD (/opt/reports/nightly.sh)\n"
            "Oct 11 02:00:01 CRON: (hatch) CMD (/opt/reports/nightly.sh)\n"
            "Oct 12 02:00:01 CRON: (hatch) CMD (/opt/reports/nightly.sh)\n"
            "$ ls -la /var/log/nightly.log\n"
            "-rw-r--r-- 1 hatch hatch 0 Sep 28 02:00 /var/log/nightly.log\n"
            "$ /opt/reports/nightly.sh; echo \"exit=$?\"; ls -la /var/log/nightly.log\n"
            "exit=0\n"
            "-rw-r--r-- 1 hatch hatch 0 Oct 12 13:50 /var/log/nightly.log\n"
            "$ crontab -l | grep nightly\n"
            "0 2 * * * /opt/reports/nightly.sh > /dev/null 2>&1"
        ),
        "config": "cron: 0 2 * * * /opt/reports/nightly.sh > /dev/null 2>&1",
        "root_causes": [
            {"id": "rc-a", "text": "The cron line redirects all output to /dev/null — failures and reports alike are discarded"},
            {"id": "rc-b", "text": "The report script is a no-op; its logic was deleted"},
            {"id": "rc-c", "text": "Log rotation is deleting nightly.log every night"},
        ],
        "fixes": [
            {"id": "fx-x", "text": "Log to a real file and alert on empty output: >> /var/log/nightly.log 2>&1 + size check"},
            {"id": "fx-y", "text": "Rewrite the report script from scratch"},
            {"id": "fx-z", "text": "Disable log rotation for nightly.log"},
        ],
        "gold": {"root_cause": "rc-a", "fix": "fx-x"},
    },
]


class CronOpsEnv(Env):
    """workgym/cron-ops-v0."""

    env_id = "workgym/cron-ops-v0"

    def __init__(self, scenario_ids=None):
        super().__init__()
        self.scenario_ids = (list(scenario_ids) if scenario_ids
                             else [s["id"] for s in _SCENARIOS])
        unknown = set(self.scenario_ids) - {s["id"] for s in _SCENARIOS}
        if unknown:
            raise ValueError(f"unknown scenario ids: {sorted(unknown)}")
        self.action_space = Text(max_length=1_000_000)
        self.observation_space = Dict({
            "scenario_id": Text(), "title": Text(), "context": Text(),
            "logs": Text(), "config": Text(),
            "root_causes": Text(), "fixes": Text(),
        })
        self._sc = None
        self._obs_cache = None

    def _sc_by_id(self, sid):
        for s in _SCENARIOS:
            if s["id"] == sid:
                return s
        raise KeyError(sid)

    def _reset(self, seed):
        sid = self._rng.choice(self.scenario_ids)
        self._sc = self._sc_by_id(sid)
        self._obs_cache = {
            "scenario_id": sid,
            "title": self._sc["title"],
            "context": self._sc["context"],
            "logs": self._sc["logs"],
            "config": self._sc["config"],
            "root_causes": [{"id": r["id"], "text": r["text"]}
                            for r in self._sc["root_causes"]],
            "fixes": [{"id": f["id"], "text": f["text"]}
                      for f in self._sc["fixes"]],
        }
        return self._obs_cache, {"scenario_id": sid}

    def _step(self, action):
        if not isinstance(action, dict) or action.get("op") != "submit":
            return self._obs_cache, 0.0, False, False, {
                "error": "action must be {'op': 'submit', 'root_cause': <id>, 'fix': <id>}"}
        gold = self._sc["gold"]
        rc_ok = action.get("root_cause") == gold["root_cause"]
        fx_ok = action.get("fix") == gold["fix"]
        reward = (0.5 if rc_ok else 0.0) + (0.5 if fx_ok else 0.0)
        return self._obs_cache, reward, True, False, {
            "scenario_id": self._sc["id"],
            "root_cause_correct": rc_ok,
            "fix_correct": fx_ok,
            "gold": gold,
        }


def _register():
    from .. import register
    register(CronOpsEnv.env_id, CronOpsEnv)


_register()
