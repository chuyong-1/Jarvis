"""Conversation history that survives a restart.

The knowledge graph already remembers FACTS ("the user prefers CSV"). What it
does not remember is the thread of a conversation, so closing the window used to
reset the assistant mid-task: "open that one" after a restart referred to
nothing, and the user had to re-explain what they were doing.

STALENESS IS THE WHOLE DESIGN PROBLEM. Restoring yesterday's turns into today's
prompt is worse than restoring nothing: the model answers a question the user
has forgotten asking. So history is restored only if the last exchange was
recent (MAX_AGE_HOURS), and only the tail of it. Older history stays on disk for
the transcript, but never re-enters the prompt.
"""
import json
import os
import time
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "conversation.json"

# Beyond this, a conversation is a different conversation.
MAX_AGE_HOURS = 12
# Kept on disk; the prompt uses a smaller tail than this.
MAX_STORED_MESSAGES = 60


class ConversationStore:
    def __init__(self, path: Path = STORE, max_age_hours: float = MAX_AGE_HOURS):
        self.path = Path(path)
        self.max_age_hours = max_age_hours
        self.saved_at: float = 0.0

    def load(self) -> list[dict]:
        """Recent history, or [] when there is none or it has gone stale."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        self.saved_at = float(data.get("saved_at") or 0)
        if not self.saved_at:
            return []
        age_hours = (time.time() - self.saved_at) / 3600
        if age_hours > self.max_age_hours:
            return []
        messages = data.get("messages") or []
        # Only well-formed chat messages; a corrupt entry would break rendering
        # of the whole prompt rather than just itself.
        #
        # 'tool' rows belong here too. They are the evidence that a past turn
        # ACTED rather than merely claimed to. Dropping them turns a restored
        # conversation into a list of bare assertions -- "Excel is opening",
        # with no call behind it -- and that is exactly what teaches the model
        # it may assert instead of act. Restoring history used to reintroduce
        # the fabrication bug on every start.
        kept = [m for m in messages
                if isinstance(m, dict)
                and m.get("role") in ("user", "assistant", "tool")
                and isinstance(m.get("content"), str)]
        # Always begin on a user message. The stored tail can be cut mid-turn,
        # and a history opening on a tool result -- a result with no call in
        # front of it -- is meaningless to the model and invalid for the chat
        # template.
        for i, m in enumerate(kept):
            if m.get("role") == "user":
                return kept[i:]
        return []

    def save(self, messages: list[dict]) -> None:
        payload = {
            "saved_at": time.time(),
            "messages": messages[-MAX_STORED_MESSAGES:],
        }
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)      # atomic: never a half-written file
        except OSError:
            pass                            # persistence is a convenience, never fatal

    def age_note(self) -> str:
        """How long ago the restored conversation happened, for the prompt."""
        if not self.saved_at:
            return ""
        minutes = (time.time() - self.saved_at) / 60
        if minutes < 2:
            return "a moment ago"
        if minutes < 90:
            return f"about {round(minutes)} minutes ago"
        hours = round(minutes / 60)
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"

    def clear(self) -> None:
        self.saved_at = 0.0
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
