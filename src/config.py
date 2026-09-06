"""Central config. ALLOWED_ROOTS is the security boundary for the whole app.

Nothing outside these directories may be read, written, listed, or executed
against -- see safety.resolve_within(), which every file tool must call before
touching the filesystem.
"""
from pathlib import Path

# An empty list means EVERY path check fails closed, which is the correct
# default: a misconfigured assistant that can reach nothing is safe, one that
# can reach everything is not.
#
# The sandbox came first so write/move/delete/run_command could be proven
# end-to-end before any real document was in reach. That is done -- 13/13
# containment attacks, 21/21 tool checks, and the confirmation flow verified
# against approve, deny, Escape and window-close -- so the everyday folders are
# now included as well.
#
# Deliberately NOT included: the whole of C:\ or D:\. Containment stops meaning
# anything once it covers everything, and D:\ would include project-jarvis
# itself, letting the assistant edit its own source and its own allowlist.
#
# Missing folders are dropped rather than failing: a machine without a Desktop,
# or with OneDrive-redirected profile folders, must still start.
_CANDIDATE_ROOTS = [
    Path(r"D:\jarvis-sandbox"),
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
]
ALLOWED_ROOTS: list[Path] = [p for p in _CANDIDATE_ROOTS if p.is_dir()]

# Which roots feed the SEMANTIC index, as opposed to which are reachable at all.
#
# These are different questions and conflating them measurably hurt answers.
# Downloads on this machine holds 229,746 entries -- mostly checked-out
# repositories -- and indexing them buried the user's own documents: "what is
# over budget" started returning a README from a cloned project instead of
# q1_budget.xlsx. Downloads is a transit folder, not a document library.
#
# Files there are still fully reachable: find_files locates them by name,
# read_file and open_file work on them. They just do not compete for relevance
# in content search. Add it back here if you want it searched by meaning too.
_NOT_INDEXED = {"downloads"}
INDEX_ROOTS: list[Path] = [p for p in ALLOWED_ROOTS if p.name.lower() not in _NOT_INDEXED]

# Destructive tools always require explicit UI confirmation, every call.
# Read-only tools may auto-execute.
# remember_fact writes only to the local knowledge graph, never to the
# filesystem, so it stays in the auto-execute tier: prompting for confirmation
# every time the assistant learns your name would make it unusable.
# Timers auto-execute: they touch no files, start no processes, and are
# trivially reversible with cancel_timer. Prompting to approve "set a timer for
# 30 seconds" would make the assistant tiresome for no safety gain.
# get_datetime and system_status read the clock and this machine's own vitals.
# They touch no files and start no processes, so they stay in the auto-execute
# tier: asking permission to know what time it is would be theatre.
READ_ONLY_TOOLS = {"search_files", "find_files", "read_file", "list_directory",
                   "remember_fact", "recall_facts",
                   "set_timer", "list_timers", "cancel_timer",
                   "get_datetime", "system_status", "list_apps"}
DESTRUCTIVE_TOOLS = {"write_file", "edit_file", "append_to_file", "create_directory",
                     "move_file", "delete_file", "run_command",
                     # Launching things is destructive-tier even though it writes
                     # nothing: it starts a process on the user's behalf.
                     "open_file", "launch_app", "open_with",
                     # draw_image writes a PNG and may then open it in an app.
                     # Both halves are destructive-tier, and one confirmation
                     # covers the whole "draw this and show it in Paint" turn
                     # rather than interrupting the user twice for one request.
                     "draw_image",
                     # open_url leaves the machine. The browser fetches the page,
                     # not the model, so there is no prompt-injection path -- but a
                     # URL can CARRY data out in its query string, so every one is
                     # shown in full and approved before it opens.
                     "open_url"}

AUDIT_DB = Path(__file__).resolve().parent.parent / "audit.db"

# run_command limits
COMMAND_TIMEOUT_SECONDS = 20
