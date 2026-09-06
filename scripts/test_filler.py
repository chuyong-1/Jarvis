"""The sign-off stripper: what it removes, what it must NOT remove, and that
the streamed view matches the returned text.

The model ends nearly every turn with "How can I assist you today?" no matter
what the system prompt says (the rule is in there; it is ignored). Spoken
aloud, that is the same sentence every time. It is removed in code instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import _hold_filler_len, strip_closing_filler  # noqa: E402

fails = 0


def check(label, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")


print("stripped — the sign-offs actually observed from the model")
check("greeting keeps the greeting",
      strip_closing_filler("Hello! How can I assist you today?"), "Hello!")
check("'How may I assist you?'",
      strip_closing_filler("I am JARVIS, your voice assistant. How may I assist you?"),
      "I am JARVIS, your voice assistant.")
check("'just let me know'",
      strip_closing_filler("You're welcome! If you have any specific requests or "
                           "need further assistance, just let me know."),
      "You're welcome!")
check("'feel free to ask'",
      strip_closing_filler("I can open files and set timers. If you need assistance "
                           "with something specific, feel free to ask!"),
      "I can open files and set timers.")
check("'Is there anything else'",
      strip_closing_filler("The timer is set. Is there anything else you need?"),
      "The timer is set.")
check("two stacked sign-offs both go",
      strip_closing_filler("Done. Let me know if you need more. How can I help further?"),
      "Done.")
check("'Would you like me to' (a sign-off wearing a question mark)",
      strip_closing_filler("I have drawn a rocket and saved it to your desktop. "
                           "Would you like me to make any changes to the drawing?"),
      "I have drawn a rocket and saved it to your desktop.")
check("'Enjoy your creation!'",
      strip_closing_filler("A smiley face has been drawn and opened in Paint. "
                           "Enjoy your creation!"),
      "A smiley face has been drawn and opened in Paint.")

print("\npreserved — content that merely LOOKS like a sign-off")
check("conditional carrying an answer",
      strip_closing_filler("The budget is over by 12%. If you need the Q1 file, it is on D:."),
      "The budget is over by 12%. If you need the Q1 file, it is on D:.")
check("plain answer untouched",
      strip_closing_filler("Dana Whitfield is drafting it, due 2 April 2026."),
      "Dana Whitfield is drafting it, due 2 April 2026.")
check("all-pleasantry reply is NOT emptied",
      strip_closing_filler("How can I assist you today?"),
      "How can I assist you today?")
check("empty stays empty", strip_closing_filler(""), "")

print("\nstreaming — the held-back tail")
check("holds a forming sign-off",
      _hold_filler_len("The timer is set. How can I"), len("The timer is set. "))
check("does not hold ordinary prose",
      _hold_filler_len("The timer is set. It ends at"), len("The timer is set. It ends at"))
check("nothing to hold mid-first-sentence",
      _hold_filler_len("The timer is"), len("The timer is"))

print("\nstream/return agreement — what was shown is a prefix of what is kept")
for text in ["Hello! How can I assist you today?",
             "The timer is set. Is there anything else?",
             "The budget is over by 12%. If you need the Q1 file, it is on D:.",
             "How can I assist you today?"]:
    final = strip_closing_filler(text)
    check(f"prefix: {text[:34]!r}...", text.startswith(final), True)
    # Whatever streaming already showed must survive the trim, or the user
    # watched text appear and then be contradicted by the logged reply.
    shown = text[:_hold_filler_len(text)].rstrip()
    check(f"stream never over-emits: {text[:24]!r}...", final.startswith(shown), True)

print(f"\n{'PASS' if not fails else 'FAIL'}: {fails} failing")
sys.exit(1 if fails else 0)
