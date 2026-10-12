"""workgym/pr-review-v0 — review a diff, find the seeded bugs.

Each episode presents one code diff. Some contain real bugs (security,
correctness, performance, error-handling); one is clean. The agent submits
findings; reward is the F1 score of (file, line) bug detection against the
hidden ground truth (line tolerance: 2 lines).

Single-step: one submit ends the episode. There is no partial credit for
vibes — a finding only counts if it points at a real seeded bug.

The seeded bugs are synthetic by design (this is a training environment, not
a published benchmark): the value is that the reward is exactly verifiable.
"""

from ..core import Env, Text, Dict

_DIFFS = [
    {
        "id": "sql-swallow",
        "title": "users.py: fetch a user by name",
        "files": {
            "src/users.py": (
                "import db\n"
                "\n"
                "def get_user(name):\n"
                "    query = f\"SELECT * FROM users WHERE name = '{name}'\"\n"
                "    try:\n"
                "        return db.query(query)\n"
                "    except Exception:\n"
                "        return None\n"
                "\n"
                "def list_users(limit=100):\n"
                "    return db.query(\"SELECT * FROM users LIMIT %s\", (limit,))\n"
            )
        },
        "diff": (
            "--- a/src/users.py\n"
            "+++ b/src/users.py\n"
            "@@ -1,6 +1,11 @@\n"
            " import db\n"
            "\n"
            " def get_user(name):\n"
            "-    return db.query(\"SELECT * FROM users WHERE name = %s\", (name,))\n"
            "+    query = f\"SELECT * FROM users WHERE name = '{name}'\"\n"
            "+    try:\n"
            "+        return db.query(query)\n"
            "+    except Exception:\n"
            "+        return None\n"
        ),
        "bugs": [
            {"file": "src/users.py", "line": 4, "category": "security",
             "note": "SQL injection: `name` interpolated into SQL via f-string"},
            {"file": "src/users.py", "line": 7, "category": "error-handling",
             "note": "bare `except Exception` swallows connection/query errors, returns None"},
        ],
    },
    {
        "id": "paginate-offbyone",
        "title": "paginate.js: slice items for a page",
        "files": {
            "src/paginate.js": (
                "export function paginate(items, page, perPage) {\n"
                "  const start = page * perPage;\n"
                "  return items.slice(start, start + perPage);\n"
                "}\n"
            )
        },
        "diff": (
            "--- a/src/paginate.js\n"
            "+++ b/src/paginate.js\n"
            "@@ -1,4 +1,4 @@\n"
            " export function paginate(items, page, perPage) {\n"
            "-  const start = (page - 1) * perPage;\n"
            "+  const start = page * perPage;\n"
            "   return items.slice(start, start + perPage);\n"
            " }\n"
        ),
        "bugs": [
            {"file": "src/paginate.js", "line": 2, "category": "correctness",
             "note": "off-by-one: page 1 skips the first perPage items; should be (page - 1) * perPage"},
        ],
    },
    {
        "id": "n-plus-one",
        "title": "reports.py: build a report for users",
        "files": {
            "src/reports.py": (
                "import db\n"
                "\n"
                "def build_report(user_ids):\n"
                "    rows = []\n"
                "    for uid in user_ids:\n"
                "        profile = db.fetch_profile(uid)\n"
                "        rows.append({\"id\": uid, \"name\": profile.name})\n"
                "    return rows\n"
            )
        },
        "diff": (
            "--- a/src/reports.py\n"
            "+++ b/src/reports.py\n"
            "@@ -1,7 +1,7 @@\n"
            " import db\n"
            "\n"
            " def build_report(user_ids):\n"
            "     rows = []\n"
            "     for uid in user_ids:\n"
            "-        profile = db.fetch_profile(uid)\n"
            "+        profile = db.fetch_profile(uid)  # one query per user\n"
            "         rows.append({\"id\": uid, \"name\": profile.name})\n"
            "     return rows\n"
        ),
        "bugs": [
            {"file": "src/reports.py", "line": 6, "category": "performance",
             "note": "N+1 queries: one db round-trip per user inside the loop; batch-fetch instead"},
        ],
    },
    {
        "id": "foreach-no-await",
        "title": "uploader.js: save a batch of items",
        "files": {
            "src/uploader.js": (
                "import db from \"./db.js\";\n"
                "import audit from \"./audit.js\";\n"
                "\n"
                "export async function saveAll(items) {\n"
                "  items.forEach(async (it) => {\n"
                "    await db.save(it);\n"
                "  });\n"
                "  audit.log(\"saved \" + items.length + \" items\");\n"
                "}\n"
            )
        },
        "diff": (
            "--- a/src/uploader.js\n"
            "+++ b/src/uploader.js\n"
            "@@ -1,8 +1,9 @@\n"
            " import db from \"./db.js\";\n"
            " import audit from \"./audit.js\";\n"
            "\n"
            " export async function saveAll(items) {\n"
            "-  for (const it of items) { await db.save(it); }\n"
            "+  items.forEach(async (it) => {\n"
            "+    await db.save(it);\n"
            "+  });\n"
            "   audit.log(\"saved \" + items.length + \" items\");\n"
            " }\n"
        ),
        "bugs": [
            {"file": "src/uploader.js", "line": 5, "category": "correctness",
             "note": "forEach does not await async callbacks: audit log fires before saves "
                     "finish and rejections go unhandled"},
        ],
    },
    {
        "id": "clean-rename",
        "title": "names.js: pure rename refactor (no bugs)",
        "files": {
            "src/names.js": (
                "export function fullName(user) {\n"
                "  return `${user.first} ${user.last}`.trim();\n"
                "}\n"
            )
        },
        "diff": (
            "--- a/src/names.js\n"
            "+++ b/src/names.js\n"
            "@@ -1,3 +1,3 @@\n"
            "-export function getFullName(u) {\n"
            "-  return (u.first + \" \" + u.last).trim();\n"
            "+export function fullName(user) {\n"
            "+  return `${user.first} ${user.last}`.trim();\n"
            " }\n"
        ),
        "bugs": [],
    },
]

_LINE_TOL = 2


class PrReviewEnv(Env):
    """workgym/pr-review-v0."""

    env_id = "workgym/pr-review-v0"

    def __init__(self, diff_ids=None):
        super().__init__()
        self.diff_ids = list(diff_ids) if diff_ids else [d["id"] for d in _DIFFS]
        unknown = set(self.diff_ids) - {d["id"] for d in _DIFFS}
        if unknown:
            raise ValueError(f"unknown diff ids: {sorted(unknown)}")
        self.action_space = Text(max_length=1_000_000)
        self.observation_space = Dict({
            "diff_id": Text(), "title": Text(), "diff": Text(), "files": Text(),
        })
        self._diff = None
        self._obs_cache = None

    def _diff_by_id(self, did):
        for d in _DIFFS:
            if d["id"] == did:
                return d
        raise KeyError(did)

    def _reset(self, seed):
        did = self._rng.choice(self.diff_ids)
        self._diff = self._diff_by_id(did)
        self._obs_cache = {
            "diff_id": did,
            "title": self._diff["title"],
            "diff": self._diff["diff"],
            "files": self._diff["files"],
        }
        return self._obs_cache, {"diff_id": did, "num_bugs": len(self._diff["bugs"])}

    def _score(self, findings):
        bugs = self._diff["bugs"]
        matched = set()
        tp = 0
        for f in findings:
            if not isinstance(f, dict):
                continue
            ffile, fline = f.get("file"), f.get("line")
            for i, b in enumerate(bugs):
                if i in matched:
                    continue
                if (b["file"] == ffile and isinstance(fline, int)
                        and abs(b["line"] - fline) <= _LINE_TOL):
                    matched.add(i)
                    tp += 1
                    break
        fp = len([f for f in findings if isinstance(f, dict)]) - tp
        fn = len(bugs) - tp
        if not bugs and not findings:
            return 1.0, {"tp": 0, "fp": 0, "fn": 0}
        if not bugs or (tp == 0 and (fp or fn)):
            # no bugs but findings -> 0; bugs but nothing matched -> 0
            if tp == 0:
                return 0.0, {"tp": tp, "fp": fp, "fn": fn}
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return f1, {"tp": tp, "fp": fp, "fn": fn}

    def _step(self, action):
        if not isinstance(action, dict) or action.get("op") != "submit":
            return self._obs_cache, 0.0, False, False, {
                "error": "action must be {'op': 'submit', 'findings': [...]}"}
        findings = action.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        reward, detail = self._score(findings)
        info = {"diff_id": self._diff["id"], "num_bugs": len(self._diff["bugs"]), **detail}
        return self._obs_cache, reward, True, False, info


def _register():
    from .. import register
    register(PrReviewEnv.env_id, PrReviewEnv)


_register()
