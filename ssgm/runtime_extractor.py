from __future__ import annotations

import re
from dataclasses import dataclass

from ssgm.models import MemoryRecord

SLOT_RULES = [
    ("meeting", ["meeting", "lab", "seminar", "appointment", "office hour", "call", "schedule"]),
    ("identity", ["name", "birthday", "degree", "graduated", "email", "phone", "address"]),
    ("preference", ["like", "prefer", "prefers", "favorite", "usually", "often", "tend to", "enjoy", "avoid"]),
    ("project", ["project", "owner", "deadline", "task", "milestone", "repo", "paper"]),
    ("location", ["office", "home", "room", "building", "campus", "city", "travel"]),
]

TIME_WORDS = {
    "today", "tonight", "tomorrow", "yesterday", "weekly", "daily",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening",
}

UPDATE_CUES = [
    "correction", "updated", "changed", "instead", "no longer", "not anymore",
    "moved", "rescheduled", "latest", "new", "now", "actually",
]

DISCOURSE_WORDS = {
    "what", "sure", "choosing", "correction", "update", "note", "revision",
    "actually", "well", "yes", "no", "okay", "ok", "thanks", "thank",
    "great", "hello", "hi", "bbq", "please",
}

PREFERENCE_CUES = ["prefer", "prefers", "favorite", "like", "likes", "enjoy", "avoid", "usually"]
SCHEDULE_CUES = ["meeting", "appointment", "call", "schedule", "every", "weekly", "daily", " at "]
STATUS_CUES = ["is ", "are ", "works", "working", "located", "in the office", "at home"]
CONTACT_CUES = ["email", "phone", "address", "contact"]
OWNERSHIP_CUES = ["owner", "my project", "our project", "deadline", "task", "repo", "paper"]


@dataclass
class RuntimeMemoryExtractor:
    def infer_slot(self, text: str) -> str:
        lowered = text.lower()
        for slot, cues in SLOT_RULES:
            if any(cue in lowered for cue in cues):
                return slot
        return "note"

    def infer_relation(self, text: str, slot: str) -> str:
        lowered = f" {text.lower()} "
        if slot == "preference" and any(cue in lowered for cue in PREFERENCE_CUES):
            return "preference"
        if slot == "meeting" and any(cue in lowered for cue in SCHEDULE_CUES):
            return "schedule"
        if slot == "location" and any(cue in lowered for cue in STATUS_CUES + ["office", "home", "room", "building", "campus", "city"]):
            return "status"
        if slot == "identity" and any(cue in lowered for cue in CONTACT_CUES):
            return "contact"
        if slot == "project" and any(cue in lowered for cue in OWNERSHIP_CUES):
            return "ownership"

        if any(cue in lowered for cue in PREFERENCE_CUES):
            return "preference"
        if any(cue in lowered for cue in SCHEDULE_CUES) or any(word in lowered for word in TIME_WORDS):
            return "schedule"
        if any(cue in lowered for cue in STATUS_CUES):
            return "status"
        if any(cue in lowered for cue in CONTACT_CUES):
            return "contact"
        if any(cue in lowered for cue in OWNERSHIP_CUES):
            return "ownership"
        if slot == "note":
            return "generic"
        return "fact"

    def infer_subject(self, text: str, tenant_id: str) -> str:
        lowered = text.lower()

        explicit_patterns = [
            r"\b(alice|bob|charlie)\b",
            r"\b(my|i|we)\b",
            r"\b(he|she|they)\b",
        ]
        for pattern in explicit_patterns:
            m = re.search(pattern, lowered)
            if m:
                subj = m.group(1)
                if subj in {"my", "i", "we"}:
                    return tenant_id
                return subj

        titlecase_candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
        for cand in titlecase_candidates:
            subj = cand.lower().replace(" ", "_")
            if subj not in DISCOURSE_WORDS:
                return subj

        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z_\-]+\b", lowered)
        for tok in tokens[:8]:
            if tok in DISCOURSE_WORDS:
                continue
            if tok in TIME_WORDS:
                continue
            if tok in {"prefer", "prefers", "like", "likes", "favorite", "meeting", "appointment", "office", "project"}:
                continue
            return tok

        return tenant_id

    def infer_time_hint(self, text: str) -> str:
        lowered = text.lower()
        for cue in TIME_WORDS:
            if cue in lowered:
                return cue.replace(" ", "_")
        if re.search(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b", lowered):
            return "clock_time"
        return "atemporal"

    def infer_update_hint(self, text: str) -> str:
        lowered = text.lower()
        for cue in UPDATE_CUES:
            if cue in lowered:
                return cue.replace(" ", "_")
        return "stable"

    def build_key(self, tenant_id: str, subject: str, slot: str, relation: str, time_hint: str, update_hint: str) -> str:
        parts = [tenant_id, subject, slot, relation]
        if time_hint != "atemporal":
            parts.append(time_hint)
        if update_hint != "stable":
            parts.append(update_hint)
        return ":".join(parts)

    def extract(self, text: str, tenant_id: str, source: str, timestamp: int) -> MemoryRecord:
        slot = self.infer_slot(text)
        relation = self.infer_relation(text, slot)
        subject = self.infer_subject(text, tenant_id)
        time_hint = self.infer_time_hint(text)
        update_hint = self.infer_update_hint(text)
        key = self.build_key(tenant_id, subject, slot, relation, time_hint, update_hint)
        return MemoryRecord(
            key=key,
            value=text,
            tenant_id=tenant_id,
            source=source,
            timestamp=timestamp,
            tags=[
                f"runtime_slot:{slot}",
                f"runtime_relation:{relation}",
                f"runtime_subject:{subject}",
                f"runtime_time:{time_hint}",
                f"runtime_update:{update_hint}",
            ],
        )
