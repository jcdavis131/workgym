"""workgym/inbox-triage-v0 — triage an inbox against a fixed rubric.

Rubric (mirrors the real Scout inbox-triage rules):
  URGENT — needs Cameron today (money, security, production down, time-bound)
  ACTION — needs a reply or decision, not today-urgent
  FYI    — worth knowing, no action needed
  NOISE  — CI/deploy notices already known, newsletters, spam

Each episode presents one inbox (10 emails). The agent submits labels;
reward is plain accuracy against the hidden gold labels.

Single-step: one submit ends the episode. Emails are fictional and
self-contained; nothing here touches a real mailbox.
"""

from ..core import Env, Text, Dict

LABELS = ("URGENT", "ACTION", "FYI", "NOISE")

_INBOXES = [
    {
        "id": "inbox-0",
        "emails": [
            {"id": "e0", "from": "bank-alerts@firstnational.example",
             "subject": "Unusual $4,812 charge on card ending 4417",
             "body": "A charge of $4,812.00 at MERIDIAN TRAVEL was attempted on your card "
                     "ending 4417 at 2:14 AM. Reply YES if authorized, NO to freeze the card.",
             "gold": "URGENT"},
            {"id": "e1", "from": "ci@builds.example",
             "subject": "[vector-hoops] build #4812 passed",
             "body": "Build #4812 (scout/live) completed successfully in 4m12s. "
                     "No action required.",
             "gold": "NOISE"},
            {"id": "e2", "from": "priya@contractor.example",
             "subject": "Question on the API rate limits doc",
             "body": "Hi! Quick question on the rate-limits page — is the 1000/hr cap per "
                     "API key or per account? No rush, whenever you get a chance.",
             "gold": "ACTION"},
            {"id": "e3", "from": "newsletter@techcrunch.example",
             "subject": "Today in startups: 14 AI raises",
             "body": "Your daily digest of startup news...",
             "gold": "NOISE"},
            {"id": "e4", "from": "nurse@wisemanfamilypractice.example",
             "subject": "Appointment confirmation — tomorrow 9:30 AM",
             "body": "Confirming your physical tomorrow at 9:30 AM. Reply C to confirm or "
                     "call to reschedule. Please arrive 15 minutes early.",
             "gold": "URGENT"},
            {"id": "e5", "from": "teammate@work.example",
             "subject": "Deploy of the billing fix is live",
             "body": "Just shipped the billing fix to prod. Monitoring dashboards look "
                     "green. Flagging in case you see anything odd.",
             "gold": "FYI"},
            {"id": "e6", "from": "landlord@oakmgt.example",
             "subject": "Lease renewal — response needed by Friday",
             "body": "Your lease renews in 60 days. The new rate is $2,850/mo (was $2,700). "
                     "Please sign the attached renewal or let us know by Friday.",
             "gold": "ACTION"},
            {"id": "e7", "from": "alerts@uptime.example",
             "subject": "RESOLVED: API latency spike",
             "body": "The latency alert from 03:12 is now resolved. Duration 11 minutes. "
                     "No customer impact detected.",
             "gold": "NOISE"},
            {"id": "e8", "from": "mom@family.example",
             "subject": "Photos from the lake weekend!",
             "body": "Attached the photos from last weekend — the kids had a blast!",
             "gold": "FYI"},
            {"id": "e9", "from": "security@github.example",
             "subject": "[GitHub] New sign-in from unrecognized device",
             "body": "A new sign-in to your GitHub account was detected from a device in "
                     "Lagos, NG just now. If this wasn't you, secure your account immediately.",
             "gold": "URGENT"},
        ],
    },
    {
        "id": "inbox-1",
        "emails": [
            {"id": "e0", "from": "ci@builds.example",
             "subject": "[eye] deploy failed: wind layer",
             "body": "Deploy failed on step 'upload-assets': 403 from the CDN. This is the "
                     "third failure this week on the same step.",
             "gold": "ACTION"},
            {"id": "e1", "from": "deals@mattressfirm.example",
             "subject": "48-HOUR SALE: 30% off everything",
             "body": "This weekend only...",
             "gold": "NOISE"},
            {"id": "e2", "from": "accountant@cpa.example",
             "subject": "Q3 estimated taxes due Sept 15",
             "body": "Reminder: your Q3 estimated payment of $18,400 is due September 15. "
                     "I can schedule it from the business account — just confirm.",
             "gold": "ACTION"},
            {"id": "e3", "from": "social@network.example",
             "subject": "You have 12 new connection requests",
             "body": "People you may know...",
             "gold": "NOISE"},
            {"id": "e4", "from": "school@briarwood.example",
             "subject": "Early pickup today — water main break",
             "body": "Due to a water main break, all students must be picked up by 1:00 PM "
                     "today. Buses are cancelled.",
             "gold": "URGENT"},
            {"id": "e5", "from": "pm@work.example",
             "subject": "Q4 roadmap draft for your review",
             "body": "Attached the Q4 roadmap draft. Would love your comments by end of "
                     "week if you have time.",
             "gold": "ACTION"},
            {"id": "e6", "from": "noreply@bank.example",
             "subject": "Your monthly statement is ready",
             "body": "Your August statement is available in online banking.",
             "gold": "FYI"},
            {"id": "e7", "from": "ci@builds.example",
             "subject": "[vector-hoops] build #4813 passed",
             "body": "Build #4813 completed successfully. No action required.",
             "gold": "NOISE"},
            {"id": "e8", "from": "neighbor@oakstreet.example",
             "subject": "Found your dog?",
             "body": "Hi — a golden retriever with a blue collar showed up in our yard. "
                     "Is Buddy missing??",
             "gold": "URGENT"},
            {"id": "e9", "from": "blog@dataweekly.example",
             "subject": "This week in data: vector DB benchmarks",
             "body": "The weekly roundup...",
             "gold": "NOISE"},
        ],
    },
    {
        "id": "inbox-2",
        "emails": [
            {"id": "e0", "from": "fraud@chase.example",
             "subject": "Action required: verify recent transactions",
             "body": "We noticed 3 transactions totaling $900 that don't match your usual "
                     "pattern. Please verify in the app within 24 hours or the card will "
                     "be temporarily restricted.",
             "gold": "URGENT"},
            {"id": "e1", "from": "hr@work.example",
             "subject": "Open enrollment ends Friday",
             "body": "Benefits open enrollment closes Friday at 5 PM. If you take no action, "
                     "your current elections roll over.",
             "gold": "ACTION"},
            {"id": "e2", "from": "ci@builds.example",
             "subject": "[dottie] nightly eval finished: 8.4/10",
             "body": "Nightly eval complete. Score 8.4, up from 8.1. Details in the dashboard.",
             "gold": "FYI"},
            {"id": "e3", "from": "promo@airline.example",
             "subject": "Double miles this fall",
             "body": "Earn double miles...",
             "gold": "NOISE"},
            {"id": "e4", "from": "contractor@devshop.example",
             "subject": "Invoice #1042 ($3,200) — net 15",
             "body": "Attaching invoice #1042 for the August sprint work ($3,200, net 15). "
                     "Let me know if anything looks off.",
             "gold": "ACTION"},
            {"id": "e5", "from": "alerts@uptime.example",
             "subject": "DOWN: checkout API (prod)",
             "body": "checkout-api is returning 500s for 100% of requests since 09:41. "
                     "Failover did not trigger.",
             "gold": "URGENT"},
            {"id": "e6", "from": "bookclub@friends.example",
             "subject": "Next book: Project Hail Mary",
             "body": "We picked the next book! Meeting Sept 28 at Maria's.",
             "gold": "FYI"},
            {"id": "e7", "from": "noreply@survey.example",
             "subject": "Rate your recent support experience",
             "body": "How did we do?",
             "gold": "NOISE"},
            {"id": "e8", "from": "dentist@smilecare.example",
             "subject": "Reminder: cleaning tomorrow 2 PM",
             "body": "Friendly reminder of your appointment tomorrow at 2:00 PM.",
             "gold": "FYI"},
            {"id": "e9", "from": "cfo@work.example",
             "subject": "Need your approval: $48k vendor contract",
             "body": "The DataDog renewal ($48k/yr) needs your sign-off by EOD to keep the "
                     "current pricing. DocuSign link inside.",
             "gold": "URGENT"},
        ],
    },
]


class InboxTriageEnv(Env):
    """workgym/inbox-triage-v0."""

    env_id = "workgym/inbox-triage-v0"

    def __init__(self, inbox_ids=None):
        super().__init__()
        self.inbox_ids = list(inbox_ids) if inbox_ids else [b["id"] for b in _INBOXES]
        unknown = set(self.inbox_ids) - {b["id"] for b in _INBOXES}
        if unknown:
            raise ValueError(f"unknown inbox ids: {sorted(unknown)}")
        self.action_space = Text(max_length=1_000_000)
        self.observation_space = Dict({
            "inbox_id": Text(), "emails": Text(), "rubric": Text(),
        })
        self._inbox = None
        self._obs_cache = None

    def _inbox_by_id(self, iid):
        for b in _INBOXES:
            if b["id"] == iid:
                return b
        raise KeyError(iid)

    def _reset(self, seed):
        iid = self._rng.choice(self.inbox_ids)
        self._inbox = self._inbox_by_id(iid)
        emails = [
            {"id": e["id"], "from": e["from"], "subject": e["subject"], "body": e["body"]}
            for e in self._inbox["emails"]
        ]
        self._obs_cache = {
            "inbox_id": iid,
            "emails": emails,
            "rubric": (
                "URGENT: needs Cameron today (money, security, production down, "
                "time-bound). ACTION: needs a reply/decision, not today-urgent. "
                "FYI: worth knowing, no action. NOISE: already-known CI/deploy "
                "notices, promos, newsletters."
            ),
        }
        return self._obs_cache, {"inbox_id": iid, "num_emails": len(emails)}

    def _step(self, action):
        if not isinstance(action, dict) or action.get("op") != "submit":
            return self._obs_cache, 0.0, False, False, {
                "error": "action must be {'op': 'submit', 'labels': {email_id: LABEL}}"}
        labels = action.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        correct, total = 0, 0
        per_label = {}
        for e in self._inbox["emails"]:
            total += 1
            gold = e["gold"]
            pred = labels.get(e["id"])
            ok = pred == gold
            correct += ok
            per_label.setdefault(gold, {"correct": 0, "total": 0})
            per_label[gold]["total"] += 1
            per_label[gold]["correct"] += ok
        reward = correct / total if total else 0.0
        return self._obs_cache, reward, True, False, {
            "inbox_id": self._inbox["id"],
            "correct": correct, "total": total,
            "per_label": per_label,
            "invalid_labels": sorted(
                set(labels.values()) - set(LABELS)) if isinstance(labels, dict) else [],
        }


def _register():
    from .. import register
    register(InboxTriageEnv.env_id, InboxTriageEnv)


_register()
