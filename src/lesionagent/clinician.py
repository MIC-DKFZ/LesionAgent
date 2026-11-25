"""ask_clinician backends. The agent works automatically; it asks only when an answer can change the read.

Request kinds:
  confirm       {"question", "lesion"?, "visit"?, "point"?}            -> {"answer": "yes"|"no"|"unsure"}
  choose        {"question", "options": [...], ...}                     -> {"answer": <option>}
  click         {"question", "visit", "near"?: [z,y,x]}                 -> {"point": [z,y,x]} or {"answer": "absent"}
  verify_point  {"question", "visit", "point": [z,y,x], "lesion"?}      -> {"answer": "accept"|"moved"|"absent", "point"}
Every answer also has "answered": bool and optional "comment". Tool handlers run in worker threads, so
`ask` may block (a host application waits for the user; a timeout returns answered=False).
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

KINDS = ("confirm", "choose", "click", "verify_point")


class Clinician:
    name = "none"

    def ask(self, request: dict) -> dict:
        return {"answered": False, "reason": "no clinician is available in this session: decide yourself and "
                "record the uncertainty in the lesion notes"}


@dataclass
class PendingQuestion:
    id: int
    request: dict
    event: threading.Event = field(default_factory=threading.Event)
    answer: dict | None = None


class InteractiveClinician(Clinician):
    """A human answers in a host application. `on_question(PendingQuestion)` is called from the tool thread;
    the host calls `answer(qid, dict)`. Blocks up to `timeout` seconds."""

    name = "interactive"
    _ids = itertools.count(1)

    def __init__(self, on_question, timeout: float = 600.0):
        self.on_question = on_question
        self.timeout = timeout
        self.pending: dict[int, PendingQuestion] = {}

    def ask(self, request: dict) -> dict:
        q = PendingQuestion(next(self._ids), request)
        self.pending[q.id] = q
        self.on_question(q)
        ok = q.event.wait(self.timeout)
        self.pending.pop(q.id, None)
        if not ok or q.answer is None:
            return {"answered": False, "reason": "the clinician did not answer in time: decide yourself"}
        return {"answered": True, **q.answer}

    def answer(self, qid: int, answer: dict) -> None:
        q = self.pending.get(qid)
        if q is not None:
            q.answer = answer
            q.event.set()

    def cancel_all(self, reason: str = "cancelled") -> None:
        for q in list(self.pending.values()):
            q.answer = None
            q.event.set()
