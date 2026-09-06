"""Wires every stage into one assistant: speech -> agent -> tools -> speech.

Owns the objects that are expensive to build (LLM, index, whisper) so the UI
can stay ignorant of them, and routes destructive-tool confirmation back to
whatever UI is attached.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import ToolCallingAgent  # noqa: E402
from apps import AppRegistry  # noqa: E402
from audit import AuditLog  # noqa: E402
from config import ALLOWED_ROOTS, AUDIT_DB, INDEX_ROOTS  # noqa: E402
from conversation import ConversationStore  # noqa: E402
from index_store import FileIndex  # noqa: E402
from knowledge import KnowledgeGraph  # noqa: E402
from speech import SpeechIO  # noqa: E402
from timers import TimerManager, format_duration  # noqa: E402
from tools import ToolExecutor  # noqa: E402
from watcher import IndexWatcher  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# Conversation turns kept in the prompt. Six messages was too short to hold a
# follow-up like "open that one" -- the referent had already scrolled out.
HISTORY_TURNS = 6


def tail_turns(messages: list[dict], turns: int) -> list[dict]:
    """The last `turns` exchanges, cut only on a user message.

    A turn is no longer two messages: it is the assistant's tool call, the
    result that came back, and the reply. Slicing a flat [-N:] would routinely
    start the history on an orphaned tool result -- a result with no call in
    front of it -- which is both meaningless to the model and invalid for the
    chat template. Cutting on user messages keeps every turn whole.
    """
    if turns <= 0:
        return []
    starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(starts) <= turns:
        return list(messages)
    return list(messages[starts[-turns]:])


class JarvisPipeline:
    def __init__(self, ui=None, load_llm=True, start_watcher=True, apps=None):
        self.ui = ui
        self._log = ui.log if ui else (lambda who, text: print(f"{who}: {text}"))
        self._state = ui.set_state if ui else (lambda s: None)
        # Token streaming is optional: a UI that cannot render a partial reply
        # (or no UI at all) gets the finished text in one go instead, and the
        # agent then skips the streaming code path entirely.
        self._stream = getattr(ui, "stream_token", None)
        self._stream_done = getattr(ui, "end_stream", None)

        self.audit = AuditLog(AUDIT_DB)
        # Indexed roots are a SUBSET of reachable roots -- see config.INDEX_ROOTS.
        self.index = FileIndex(INDEX_ROOTS)
        # Loaded from apps.json, not rescanned: discovery spawns PowerShell and
        # the user's choices are what matter here, not what is newly installed.
        # The APPS tab rescans on demand.
        self.apps = apps if apps is not None else AppRegistry()
        self.kg = KnowledgeGraph()
        self.speech = SpeechIO(
            on_state=self._state,
            on_level=(ui.set_level if ui and hasattr(ui, "set_level") else None))

        self.timers = TimerManager(on_fire=self._timer_fired)

        self.executor = ToolExecutor(
            ALLOWED_ROOTS, self.audit, index=self.index,
            # With no UI attached there is nothing that can legitimately approve
            # a destructive action, so refuse rather than silently auto-approve.
            confirm_fn=(ui.confirm if ui else (lambda *a: False)),
            kg=self.kg,
            timers=self.timers,
            apps=self.apps,
        )

        self.llm = None
        self.agent = None
        if load_llm:
            from llama_cpp import Llama
            self.llm = Llama(model_path=str(MODEL_PATH), n_gpu_layers=-1,
                             n_ctx=8192, verbose=False)
            self.agent = ToolCallingAgent(self.llm, self.executor)

        self.watcher = None
        if start_watcher:
            self.watcher = IndexWatcher(
                self.index, INDEX_ROOTS,
                on_event=lambda p, action: self._log("system", f"index: {p.name} {action}"))
            self.watcher.start()

        # Picked up where the last session left off, if it was recent enough to
        # still be the same conversation -- see conversation.py.
        self.conversation = ConversationStore()
        self.history: list[dict] = self.conversation.load()
        if self.history:
            # Count user messages, not pairs: a turn is now the call, its
            # result and the reply, so halving the message count over-reports.
            exchanges = sum(1 for m in self.history if m.get("role") == "user")
            self._log("system", f"Resumed {exchanges} earlier exchange(s) "
                                f"from {self.conversation.age_note()}.")
        # One llama.cpp context, several callers: the voice loop, the typed-input
        # thread, and a firing timer can all arrive at once. llama_cpp is NOT
        # thread-safe -- two concurrent generations share one KV cache and
        # corrupt each other's output -- and self.history would interleave. Every
        # turn takes this lock, so turns queue instead of colliding.
        self._turn_lock = threading.Lock()

    # ---- text turn -------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a turn is generating. The UI uses this to queue, not race."""
        return self._turn_lock.locked()

    def ask(self, text: str, on_token=None) -> str:
        if self.agent is None:
            return "(no model loaded)"
        with self._turn_lock:
            reply = self.agent.chat(text, history=tail_turns(self.history, HISTORY_TURNS),
                                    on_token=on_token)
            for call in self.agent.last_tool_calls:
                self._log("tool", f"{call['name']}({call['arguments']})")
            # Keep what the turn DID, not only what it said. History of bare
            # assertions ("Excel is opening") with no call attached teaches the
            # model that asserting IS acting, and it stops calling tools at all.
            # Reproduced exactly: cold, "open excel" calls launch_app every
            # time; replayed against a history of evidence-free assertions, the
            # same words called nothing and invented the outcome.
            trace = self.agent.last_turn_trace or [
                {"role": "assistant", "content": reply, "tool_calls": []}]
            self.history += [{"role": "user", "content": text}] + trace
            # Bounded by TURNS, not messages: one turn is now several messages
            # (the call, its result, the reply) and cutting between them would
            # leave a tool result with no call in front of it.
            self.history = tail_turns(self.history, HISTORY_TURNS * 2)
            self.conversation.save(self.history)
            return reply

    def _end_stream(self, reply: str) -> None:
        """Close a streamed reply, or log it whole when streaming is unavailable."""
        if self._stream_done is not None:
            self._stream_done()
        elif reply:
            self._log("jarvis", reply)

    def speak(self, text: str) -> None:
        if not self.speech.speak(text) and self.speech.tts_name() == "OFFLINE":
            self._log("system", "(no voice output available on this machine)")

    # ---- timers ----------------------------------------------------------

    def _timer_fired(self, timer) -> None:
        """Called from the timer's own thread when a countdown completes."""
        what = f" — {timer.label}" if timer.label else ""
        message = f"Timer finished ({format_duration(timer.seconds)}){what}."
        self._log("jarvis", "⏰ " + message)
        # An audible cue matters: the point of a timer is that you are NOT
        # watching the window. Falls back silently if the console beep is
        # unavailable (some VMs and audio configurations).
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:  # noqa: BLE001
            pass
        self.speak(message)

    # ---- voice loop ------------------------------------------------------

    def run_voice_loop(self, stop_event: threading.Event) -> None:
        self._log("system", "Listening for the wake word “Hey Jarvis”…")
        while not stop_event.is_set():
            if not self.speech.wait_for_wake_word(stop_event):
                break
            self._log("system", "Wake word detected.")

            audio = self.speech.record_utterance(stop_event)
            if stop_event.is_set():
                break

            text = self.speech.transcribe(audio)
            if not text:
                self._log("system", "(heard nothing)")
                continue
            self._log("user", text)

            self._state("thinking")
            try:
                reply = self.ask(text, on_token=self._stream)
            except Exception as exc:  # noqa: BLE001 - a bad turn must not end the loop
                self._log("denied", f"{type(exc).__name__}: {exc}")
                self._end_stream("")
                continue
            self._end_stream(reply)
            self.speak(reply)
            self._state("listening")

    def shutdown(self):
        # Cut any reply in progress first: sd.wait() on a 20-second utterance
        # would otherwise hold the window open long after the user closed it.
        self.speech.stop_speaking()
        self.timers.cancel_all()
        if self.watcher:
            self.watcher.stop()
        self.audit.close()
