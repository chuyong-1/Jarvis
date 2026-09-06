"""Stage 4: the tool-calling loop.

Rather than few-shot prompting a JSON format of our own invention, this renders
Qwen2.5's OWN chat template -- the one embedded in the GGUF metadata, which the
model was actually trained on. It takes a `tools` list and teaches the model to
emit:

    <tool_call>
    {"name": "read_file", "arguments": {"path": "..."}}
    </tool_call>

Using the native format matters: a model asked to produce a schema it was
trained on is dramatically more reliable than one coaxed into an ad-hoc format
by examples, and it costs no prompt tokens for the examples themselves.

The loop is deliberately bounded (MAX_ROUNDS). A model that loops calling the
same tool forever is a real failure mode, not a hypothetical one.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import TOOL_SCHEMA  # noqa: E402

MAX_ROUNDS = 6
MAX_TOKENS = 768           # 512 truncated longer answers mid-sentence
# How much of a tool result is kept in the history a LATER turn sees. Enough to
# show what the call returned; not the whole of a file search, which would fill
# the context with text already acted on.
TRACE_RESULT_CHARS = 220
TOOL_TAG = "<tool_call>"

# Capture the whole tag body, not a brace-delimited slice. Qwen's OWN embedded
# template tells the model to emit doubled braces:
#     <tool_call>\n{{"name": ..., "arguments": ...}}\n</tool_call>
# (the template author wrote {{...}} inside a Jinja string literal, and Jinja
# does not unescape braces there -- HF transformers renders it the same way, so
# this is genuinely the format the model produces). A naive r"\{.*?\}" matches
# an unbalanced prefix of that and silently drops EVERY tool call.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Truncated final call: an opening tag with no closing tag before end-of-text.
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*(.*)\Z", re.DOTALL)

SYSTEM_PROMPT_TEMPLATE = """You are JARVIS, a voice assistant running entirely on \
this machine. No cloud, no network. You are capable, direct, and unflappable.

The current date and time is {now}. Trust this over any date in your training
data, and over dates written inside documents.

You may access ONLY these directories:
{roots}

How you speak:
- Your replies are read ALOUD. Write for the ear: no markdown, no bullet lists,
  no raw file paths unless the user asked for the path itself.
- One to three sentences by default. Expand only when asked for detail.
- Greetings and general knowledge need no tools. Just answer.
- But THIS machine, THESE files, and this user are never general knowledge, no
  matter how casually the question is put. "How is the machine holding up?" is
  a system_status call. You have no other way to know those numbers, and
  guessing them is the worst thing you can do.

How you act:
- When the user asks for something to be DONE, emit the tool call that does it.
  A sentence describing an action is not the action. Never say a file was
  opened, deleted, created, moved, or changed unless a tool has just returned
  success for it.
- Always use ABSOLUTE paths beginning with one of the directories above. Never
  use relative paths like "reports" -- write "{example}" instead.
- Never ask the user which directory or file to look in. You already know the
  roots: search or list them yourself and find it. Asking is a failure.
- When the user names a document loosely -- "my meeting notes", "the budget",
  "that contract" -- do NOT ask which file they mean. Call search_files with
  those very words, or list_directory on a root, and work out which file it is.
- NEVER guess a filename. Before opening, reading, editing, moving or deleting
  anything, you must have seen its real path in a tool result. "my meeting
  notes" is not a filename; meeting_notes.txt is. Call find_files first.
- Choosing a lookup tool, and this matters:
  * To ANSWER A QUESTION about what documents say -- who, what, when, why, how
    much -- call search_files. It already searches the contents of every folder,
    so it is the first and usually the only call. Do not run find_files first.
  * To GET A PATH so you can act on a file -- open, edit, move, delete, or read
    it whole -- call find_files, then act on the exact path it returns.
  * list_directory shows one folder. read_file needs an exact path.
- To change a file, prefer edit_file (exact snippet -> replacement) over
  write_file, which replaces the whole file and loses anything you omit.
- Never invent file contents, paths, numbers, or dates. Read them, or say you
  could not find them.
- Never write out what a tool MIGHT return. Text that looks like tool output
  but came from you is a fabrication. Call the tool and use its real result, or
  say you do not know.
- If a tool result tells you the correct path, or asks you to retry with one,
  do it immediately. Never hand a path back to the user to confirm when a tool
  has just told you what it is -- that is asking them to do your work.
- If a tool is DENIED, or fails for any other reason, say so plainly and stop.
  Do not repeat an identical failing call, and never report success that a tool
  did not report.
- remember_fact when the user tells you something durable about themselves,
  their team, or their projects."""


APPS_BLOCK = """

Applications the user has authorised you to open ({count}):
{apps}
Use launch_app to start one, or open_with to open a specific file in one. These
are the only applications available; anything else must be refused, not guessed."""

# Above this many, the list costs more prompt than it saves and the model is
# told to call list_apps instead.
MAX_INLINE_APPS = 40


def build_system_prompt(roots, now: str | None = None, apps=None) -> str:
    root_list = "\n".join(f"  {r}" for r in roots)
    example = str(Path(roots[0]) / "reports") if roots else "D:\\example\\reports"
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        roots=root_list, example=example,
        now=now or datetime.now().strftime("%A, %d %B %Y, %H:%M"))

    # The enabled apps go straight into the prompt rather than behind a tool
    # call. "Open this in VS Code" should not cost a round trip to discover that
    # VS Code exists, and a model that can see the list stops inventing names
    # that are not on it.
    if apps is not None:
        allowed = apps.allowed()
        if not allowed:
            prompt += ("\n\nThe user has not authorised any applications yet. If asked to "
                       "open or launch one, say so and point them at the APPS tab.")
        elif len(allowed) <= MAX_INLINE_APPS:
            prompt += APPS_BLOCK.format(count=len(allowed),
                                        apps="\n".join(f"  {a.name}" for a in allowed))
        else:
            prompt += (f"\n\nThe user has authorised {len(allowed)} applications. Call "
                       f"list_apps to see them before launching anything.")
    return prompt


def _safe_emit_len(buf: str) -> int:
    """How much of `buf` can be shown without risking a leaked tool-call tag.

    Streaming must never put "<tool_call>" and its JSON in front of the user.
    Two cases, and an early version handled only the first:

      1. The turn OPENS with a call -- nothing is emitted at all.
      2. The model writes a sentence FIRST and then calls a tool. That prose is
         worth showing, but everything from the tag onward is not.

    So: emit up to the tag if one has appeared, and otherwise hold back the
    longest suffix that is still a viable prefix of the tag, because the next
    token may complete it.
    """
    idx = buf.find(TOOL_TAG)
    if idx != -1:
        return idx
    for n in range(min(len(TOOL_TAG) - 1, len(buf)), 0, -1):
        if buf.endswith(TOOL_TAG[:n]):
            return len(buf) - n
    return len(buf)


# The sign-off the model will not stop writing. Asking it not to does not work:
# the rule is in the system prompt, verified present, and a 7B still ends turn
# after turn with "How can I assist you today?" -- naming the exact phrases in a
# prohibition may even prime them. Spoken aloud it is the same sentence every
# time, which is what makes the assistant feel dim. So it is removed in code,
# where it is a certainty rather than a request.
_FILLER_OPENER = re.compile(
    r"""(?ix) ^ \s* (?:
        (?:so\s+)? how \s+ (?:can|may) \s+ i \s+ (?:assist|help)
      | is \s+ there \s+ anything \s+ (?:else|further|more)
      | (?:just\s+)? let \s+ me \s+ know
      | feel \s+ free \s+ to
      | don'? t \s+ hesitate
      # What follows "if you need" is what separates a sign-off from a real
      # conditional: "if you need ANYTHING else" / "...ASSISTANCE" is filler,
      # "if you need THE Q1 FILE, it is on D:" is an answer, and stripping the
      # second would lose content. So the abstract objects are listed.
      | if \s+ you \s+ (?:have|need) \s+
            (?: any | assistance | help | further | more \s+ (?:help|info) )
      | if \s+ you'? (?:d|ll) \s+ like
      | if \s+ you \s+ would \s+ like
      | i'? m \s+ here \s+ (?:to \s+ help | if \s+ you)
      | what \s+ else \s+ can \s+ i
      # "Would you like me to make any changes?" -- the same sign-off wearing a
      # question mark, seen after the drawing tool landed.
      | would \s+ you \s+ like \s+ (?:me \s+ to | any | to \s+ (?:make|add))
      | do \s+ you \s+ need \s+ (?:assistance|anything|help)
      | enjoy \s+ your
    )""")

# End of a sentence, so the tail can be examined on its own.
_SENTENCE_END = re.compile(r"[.!?]['\")\]]?\s+")


def _split_sentences(text: str) -> list[str]:
    """Sentences with their terminators and trailing space kept, so that
    "".join(_split_sentences(t)) == t exactly."""
    out, last = [], 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return out


def strip_closing_filler(text: str) -> str:
    """Drop trailing offer-of-further-help sentences. Returns a PREFIX of
    `text` (right-trimmed only), which is what lets the streaming path treat
    already-emitted characters as still valid.

    A reply that is nothing BUT filler keeps its last sentence: answering "hi"
    with silence is worse than answering it with a greeting.
    """
    sents = _split_sentences(text)
    kept = len(sents)
    while kept > 0 and _FILLER_OPENER.match(sents[kept - 1]):
        kept -= 1
    if kept == 0:                      # the whole reply was pleasantry
        return text.rstrip()
    return "".join(sents[:kept]).rstrip()


# The same openers as literals, for the streaming half of the problem. A regex
# answers "is this a sign-off?"; streaming must answer "could this STILL become
# one?", because at the moment "How can I" arrives, nothing has matched yet --
# and if it is emitted then, the trim afterwards cannot take it back. Keep in
# step with _FILLER_OPENER above; test_filler.py checks both together.
_OPENER_PREFIXES = (
    "how can i assist", "how can i help", "how may i assist", "how may i help",
    "so how can i help", "is there anything else", "is there anything further",
    "is there anything more", "let me know", "just let me know", "feel free to",
    "don't hesitate", "dont hesitate", "if you have any", "if you need any",
    "if you need assistance", "if you need help", "if you need further",
    "if you'd like", "if you'll like", "if you would like", "i'm here to help",
    "im here to help", "i'm here if you", "what else can i",
    "would you like me to", "would you like any", "would you like to make",
    "would you like to add", "do you need assistance", "do you need anything",
    "do you need help", "enjoy your",
)


def _hold_filler_len(buf: str) -> int:
    """How much of `buf` may stream out yet.

    Mirrors _safe_emit_len: hold back the trailing sentence while it is still a
    viable prefix of a sign-off, because the rest of it has not arrived. Prose
    that cannot become one streams with no delay; a false hold costs only the
    few tokens until the tail stops matching, and then flushes at once.
    """
    starts = 0
    for m in _SENTENCE_END.finditer(buf):
        starts = m.end()
    tail = " ".join(buf[starts:].split()).lower()
    if not tail:
        return len(buf)
    if any(op.startswith(tail) for op in _OPENER_PREFIXES) or _FILLER_OPENER.match(tail):
        return starts
    return len(buf)


class ToolCallingAgent:
    def __init__(self, llm, executor, system_prompt: str | None = None, verbose: bool = False):
        self.llm = llm
        self.executor = executor
        # The model cannot emit valid absolute paths unless it is told what the
        # allowed roots actually are. A caller-supplied prompt is used verbatim;
        # otherwise it is rebuilt each turn so the clock in it stays current --
        # an assistant left running overnight must not still believe it is
        # yesterday.
        self._fixed_prompt = system_prompt
        self.system_prompt = system_prompt or build_system_prompt(
            executor.roots, apps=getattr(executor, "apps", None))
        self.verbose = verbose
        self.template = jinja2.Template(
            llm.metadata["tokenizer.chat_template"],
            undefined=jinja2.StrictUndefined,
        )
        self.last_tool_calls: list[dict] = []

    # ---- prompt rendering ----------------------------------------------

    def render(self, messages: list[dict]) -> str:
        return self.template.render(
            messages=messages,
            tools=[t["function"] for t in TOOL_SCHEMA],
            add_generation_prompt=True,
        )

    @staticmethod
    def _loads_tolerant(blob: str) -> dict | None:
        """Parse one tool-call body, coping with the shapes models actually emit."""
        blob = blob.strip().strip("`")
        if blob.startswith("json"):          # ```json fences
            blob = blob[4:].strip()

        attempts = [blob]
        # Qwen's own template tells it to wrap the call in DOUBLED braces, and it
        # is not consistent about closing them: the same prompt yields
        #   {{"name": ..., "arguments": {...}}}   (balanced once de-doubled)
        # and sometimes
        #   {{"name": ..., "arguments": {...}}    (one closing brace short)
        # Both are common, so try trimming either end independently rather than
        # assuming a matched pair. Getting this wrong drops the tool call
        # silently and looks like the model simply declining to act.
        if blob.startswith("{{"):
            if blob.endswith("}}"):
                attempts.append(blob[1:-1])   # symmetric de-double
            attempts.append(blob[1:])         # extra '{' only
        if blob.endswith("}}"):
            attempts.append(blob[:-1])        # extra '}' only

        # Balanced-brace scan as a last resort (handles trailing prose). Run it
        # from the first AND second character so a doubled prefix cannot make
        # the scan never reach depth zero.
        for start in {blob.find("{"), blob.find("{") + 1}:
            if start < 0 or start >= len(blob) or blob[start] != "{":
                continue
            depth = 0
            for i in range(start, len(blob)):
                if blob[i] == "{":
                    depth += 1
                elif blob[i] == "}":
                    depth -= 1
                    if depth == 0:
                        attempts.append(blob[start:i + 1])
                        break

        for attempt in attempts:
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "name" in obj:
                return obj
        return None

    @classmethod
    def parse_tool_calls(cls, text: str) -> list[dict]:
        """Extract tool calls, tolerating malformed JSON without crashing."""
        bodies = TOOL_CALL_RE.findall(text)
        if not bodies:
            m = TOOL_CALL_OPEN_RE.search(text)
            if m:
                bodies = [m.group(1)]

        calls = []
        for raw in bodies:
            obj = cls._loads_tolerant(raw)
            if obj is None:
                continue
            args = obj.get("arguments", {})
            # Some models emit arguments as a JSON *string* rather than an object.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        return calls

    # ---- main loop ------------------------------------------------------

    def _generate(self, prompt: str, on_token=None) -> str:
        """One completion. Streams to on_token when it is safe to do so.

        Tool calls are never streamed: the user should see the RESULT of a
        lookup, not the model's JSON. Prose is streamed from the first token,
        which is the difference between an assistant that feels alive and one
        that stares back for six seconds.
        """
        kwargs = dict(max_tokens=MAX_TOKENS,
                      temperature=0.0,      # determinism matters for tool selection
                      repeat_penalty=1.05,
                      stop=["<|im_end|>"])
        if on_token is None:
            return self.llm(prompt, **kwargs)["choices"][0]["text"].strip()

        buf, emitted, sealed = "", 0, False
        for chunk in self.llm(prompt, stream=True, **kwargs):
            piece = chunk["choices"][0]["text"]
            if not piece:
                continue
            buf += piece
            if sealed:                        # a tool call started; show no more
                continue
            # Two independent reasons to hold text back: a tool tag may be
            # forming, or a sign-off may be. Whichever wants to show less wins.
            end = min(_safe_emit_len(buf), _hold_filler_len(buf))
            sealed = TOOL_TAG in buf
            if end > emitted:
                on_token(buf[emitted:end])
                emitted = end

        # Generation is over, so a held-back tail is now known in full: emit
        # whatever survives the strip. This is what rescues the all-pleasantry
        # reply -- "hi" answered with nothing but "How can I assist you today?"
        # was held back in full, and strip_closing_filler keeps it rather than
        # leaving the user staring at an empty turn.
        if not sealed:
            final = strip_closing_filler(buf)
            if len(final) > emitted:
                on_token(final[emitted:])
        return buf.strip()

    def chat(self, user_message: str, history: list[dict] | None = None,
             on_token=None) -> str:
        system = self._fixed_prompt or build_system_prompt(
            self.executor.roots, apps=getattr(self.executor, "apps", None))
        self.system_prompt = system

        # Hybrid retrieval, half one: pull relevant long-term facts into context
        # BEFORE the model runs. Relying only on the recall_facts tool means the
        # model must first suspect that it knows something -- but the whole point
        # of memory is answering without being asked to look. The vector store
        # (half two) is reached on demand through search_files, because file
        # content is far too large to inject speculatively.
        kg = getattr(self.executor, "kg", None)
        self.last_injected_facts = []
        if kg is not None:
            facts = kg.context_for(user_message, limit=8)
            self.last_injected_facts = facts
            if facts:
                lines = "\n".join(f"- {f['text']}" for f in facts)
                system += ("\n\nThings you already know about this user (from memory):\n"
                           f"{lines}\n"
                           "Use these if relevant. Do not call recall_facts for anything "
                           "already listed here.")

        messages = [{"role": "system", "content": system}]
        messages += history or []
        messages.append({"role": "user", "content": user_message})

        self.last_tool_calls = []
        # What actually happened this turn, for the caller to keep as history.
        # Storing only the prose reply is what taught the model to fabricate:
        # every past turn then looks like an assistant ASSERTING "Excel is
        # opening" with no call attached, so it copies that and stops calling
        # tools. Replayed with a history of evidence-free assertions, "open
        # excel" made zero tool calls; cold, it calls launch_app every time.
        self.last_turn_trace: list[dict] = []

        for _ in range(MAX_ROUNDS):
            prompt = self.render(messages)
            text = self._generate(prompt, on_token=on_token)
            if self.verbose:
                print(f"\n[model] {text}\n")

            calls = self.parse_tool_calls(text)
            if not calls:
                # The prose answer, and the only path the user ever hears. The
                # streamed view was trimmed the same way, so what was shown and
                # what is returned (logged, spoken, saved to history) agree.
                final = strip_closing_filler(text)
                self.last_turn_trace.append({"role": "assistant", "content": final,
                                             "tool_calls": []})
                return final

            # Qwen's template iterates message.tool_calls on assistant turns.
            # The raw text already contains the <tool_call> tags, so pass an
            # empty list and let `content` carry them through verbatim.
            messages.append({"role": "assistant", "content": text, "tool_calls": []})
            self.last_turn_trace.append({"role": "assistant", "content": text,
                                         "tool_calls": []})
            for call in calls:
                self.last_tool_calls.append(call)
                result = self.executor.execute(call["name"], call["arguments"])
                if self.verbose:
                    print(f"[tool] {call['name']}({call['arguments']}) -> "
                          f"{'ok' if result.ok else 'FAILED'}: {result.output[:120]}")
                output = result.output if result.ok else f"ERROR: {result.output}"
                messages.append({"role": "tool", "content": output})
                # The trace is what LATER turns will see, so the result is kept
                # short: what matters next turn is that the call happened and
                # whether it worked, not the full body of a file search.
                self.last_turn_trace.append({
                    "role": "tool",
                    "content": output if len(output) <= TRACE_RESULT_CHARS
                    else output[:TRACE_RESULT_CHARS] + " …(truncated)",
                })

        return "(stopped: exceeded the maximum number of tool rounds)"
