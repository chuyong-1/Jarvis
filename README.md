<p align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,40:062A4A,100:1EA7FF&height=180&section=header&text=Jarvis&fontSize=50&fontColor=FFFFFF&fontAlignY=36&animation=fadeIn&desc=Fully%20offline%20voice%20assistant%20%20%C2%B7%20%20no%20cloud%20APIs&descSize=16&descAlignY=58&descColor=D6EEFF" alt="Jarvis" />
</p>

# Jarvis — fully local, offline voice assistant

No cloud APIs. No Ollama. The LLM runs in-process via `llama-cpp-python` against a
local GGUF. Everything else (STT, wake word, TTS, embeddings) is local too.

Built and measured on: Windows 11, i7-12650H, 16GB RAM, RTX 4050 Laptop (6GB VRAM).

---

## What works

| Stage | Component | Status | Verify with |
|---|---|---|---|
| 1 | Local LLM (Qwen2.5-7B Q4_K_M) | ✅ 34 tok/s, 29/29 layers on GPU | `scripts/smoke_test.py` |
| 2.1 | STT (faster-whisper `base.en`, CPU) | ✅ RTF 0.25 | `scripts/test_stt.py` |
| 2.2 | Wake word (openWakeWord `hey_jarvis`) | ✅ 0.998 vs 0.004 ambient | `scripts/test_wakeword.py` |
| 2.3 | TTS (Windows SAPI, or Piper) | ✅ RTF 0.26, speaks with nothing downloaded | `scripts/test_voice_out.py` |
| 3 | Path containment + audit | ✅ 13/13 attacks, 21/21 tool checks | `scripts/test_safety.py`, `test_tools.py` |
| 3 | Indexer + watcher (Chroma) | ✅ hybrid retrieval | `scripts/test_index.py` |
| 4 | Tool calling | ✅ 100% over 3 runs | `scripts/test_agent.py` |
| 4 | Behaviour (uses tools when it must, invents nothing) | ✅ 27/27 over 3 runs | `scripts/test_behaviour.py --runs 3` |
| 4 | Reply hygiene (no repeated sign-offs) | ✅ 24/24 | `scripts/test_filler.py` |
| 4 | Drawing (`draw_image` → PNG → Paint) | ✅ 24/24 | `scripts/test_draw.py` |
| 4 | App control (user-curated allowlist, open-in-app) | ✅ 30/30 | `scripts/test_apps.py` |
| 5 | Conversation memory across restarts | ✅ 16/16 | `scripts/test_memory.py` |
| 5 | History records actions, not just prose | ✅ 22/22 | `scripts/test_history_trace.py` |
| 4 | Local actions (edit/append/mkdir/open/launch) | ✅ 18/18 + live model | `scripts/test_edit_tools.py`, `test_local_actions.py` |
| 4 | Finding + opening files by name | ✅ 16/16 | `scripts/test_find_open.py` |
| 5 | Knowledge graph + hybrid | ✅ 16/16 | `scripts/test_knowledge.py` |
| 6 | App shell (tkinter HUD + tray) | ✅ 25/25 threading checks | `scripts/test_app.py` |
| — | PyInstaller bundle | ✅ 2.08GB, `llama_cpp` loads frozen | `dist\Jarvis\Jarvis.exe --check` |

Run it: `venv\Scripts\python.exe main.py`
(`--ui-only` skips model loading, `--check` verifies native libs and exits)

## Interface

HUD-styled: an animated arc-reactor status ring inside a hexagonal frame, whose
colour and rotation speed are driven by state (standby / listening / recording /
processing / speaking / fault), with a radar sweep, orbiting markers, and inner
spectrum bars showing **live mic amplitude** while recording. State changes *ease*
between colours over ~200ms rather than snapping, because a hard cut reads as a
glitch. Cyan-on-black, monospace, colour-coded transcript, system tray icon.

Three tabs: **TRANSCRIPT** (replies stream in token by token, with quick-action
chips and shell-style Up/Down command history), **AUDIT LOG** (every tool call),
and **FILES** — a browser over the allowed roots that previews any file in-app and
opens it externally, so the commonest request does not need a round trip through a
7B model. Live telemetry (VRAM, RAM, generation rate, uptime, tool counts) refreshes
once a second. Shortcuts: `Enter` send, `Ctrl+Enter` engage the microphone, `Esc`
stop speaking, `Ctrl+L` clear.

Two tkinter obstacles worth knowing about, both fixed here:

* Widget colours require the `clam` ttk theme — the native Windows theme silently
  ignores most colour options. clam then draws Entry and Scrollbar as 3-D bevels,
  so `fieldbackground` alone leaves a **white border**: `lightcolor`, `darkcolor`
  and `bordercolor` must be set too.
* **DPI.** Without declaring DPI awareness the process renders at 96dpi and Windows
  bitmap-stretches the result — 125% on this display. A HUD is drawn almost entirely
  from one-pixel lines, and every one of them comes out soft. `theme.setup_dpi()`
  runs before the Tk root exists; Tk then scales point-sized fonts itself and only
  explicit pixel dimensions need `app.px()`.

> This is HUD-*styled*, not a replica. The film's interface is a volumetric 3-D
> hologram; tkinter draws 2-D canvas primitives. What you get is the palette,
> the concentric animated rings, angular framing and telemetry — not depth,
> particles, or transparency.

---

## Setup

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

:: llama-cpp-python needs the CUDA wheel index (NOT on PyPI)
venv\Scripts\python.exe -m pip install llama-cpp-python ^
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

:: ...and the CUDA 12 runtime, which that wheel does NOT bundle
venv\Scripts\python.exe -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

venv\Scripts\python.exe scripts\download_model.py
```

### ⚠️ The one non-obvious step

`llama-cpp-python`'s CUDA wheel ships `ggml-cuda.dll` (703MB of kernels) but **not**
the CUDA runtime. Without `cudart64_12.dll` / `cublas64_12.dll` on the DLL search
path, importing it fails with a misleading error naming the *wrong* file:

```
Could not find module '...\llama_cpp\lib\llama.dll' (or one of its dependencies)
```

`llama.dll` is present. Its *dependencies* are not.

`os.add_dll_directory()` alone does **not** fix this: `llama_cpp` loads the DLL with
`winmode=ctypes.RTLD_GLOBAL`, which is `0` on Windows, selecting the legacy
`LoadLibraryExW` search order that ignores registered DLL directories. That order
*does* search `PATH`. Hence `venv/Lib/site-packages/_cuda_dll_path.pth`:

```python
import os, glob, sys; _d = [p for p in glob.glob(os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', '*', 'bin')) if os.path.isdir(p)]; [os.add_dll_directory(p) for p in _d]; os.environ['PATH'] = os.pathsep.join(_d) + os.pathsep + os.environ.get('PATH', '')
```

**Rebuilding the venv deletes this file and breaks `llama_cpp`.** Recreate it.
The packaged build handles the same problem via `packaging/runtime_hook_cuda.py`,
because frozen apps do not process `.pth` files at all.

### Voice output — nothing to download

Two local backends, probed in order at startup and cached:

| Backend | When it is used | Notes |
|---|---|---|
| **Piper** | if `piper\piper.exe` + the voice exist | better voice, manual download |
| **Windows SAPI** | otherwise | ships with the OS, RTF 0.26 measured here |

Piper was originally the only backend, which meant a fresh machine was **silent** —
the one failure mode a voice assistant cannot have. SAPI is driven through
PowerShell's `System.Speech`, synthesising to a wav that is then played through the
same sounddevice path as Piper, so both share one interruptible player.

> **The text spoken is model output, so it is never interpolated into the PowerShell
> command line.** It is written to a UTF-8 file the script reads back; every path in
> the command is an app-owned constant. Passing generated text through a shell
> command line would be an injection hole regardless of how safe the model seems.

Replies are normalised before being spoken (`speech._speakable`): markdown stripped,
long paths reduced to filenames — `D:\jarvis-sandbox\notes\ideas.md` is read as
"ideas.md" rather than spelled out — and runaway replies truncated at a sentence
boundary. The transcript still shows the full text.

Piper, if you want the better voice:

1. `piper_windows_amd64.zip` from https://github.com/rhasspy/piper/releases →
   extract the **whole folder** (exe + DLLs + `espeak-ng-data\`) to `piper\`
2. `en_US-lessac-medium.onnx` **and** `.onnx.json` from
   https://huggingface.co/rhasspy/piper-voices (`en/en_US/lessac/medium/`) → `models\piper\`

`scripts/test_voice_out.py` reports which backend is live and verifies it produces
audio; `scripts/test_tts.py` is the Piper-specific check.

---

## Hardware decisions, with measurements

### The GPU belongs to the LLM

| n_ctx | VRAM | headroom | tok/s |
|---|---|---|---|
| 4096 | 4803 MiB | 1338 | 34.1 |
| **8192** | **5217 MiB** | **924** | **34.4** ← default |
| 16384 | 5893 MiB | 248 | 34.0 |
| 32768 | 5893 MiB *(capped)* | 248 | 29.8 ⚠️ |

**32768 does not error — it silently degrades.** VRAM stops climbing and the excess
KV cache spills into shared system RAM via WDDM. If generation feels slow, check
`nvidia-smi` before blaming the model.

Whisper therefore runs on **CPU**. Putting it on GPU frees ~700MB by evicting ~5
llama.cpp layers, and partial offload is brutally nonlinear: ~34 tok/s collapses to
~10-15, on *every token of every reply*, to save ~1.3s of one-shot STT. STT and
generation are sequential anyway, so sharing the GPU buys no parallelism.

### Whisper model size (measured, CPU int8, 5.31s clean audio)

| model | beam | RTF |
|---|---|---|
| **base.en** | **1** | **0.25** ← default |
| base.en | 5 | 0.28 |
| small.en | 1 | 0.81 |
| small.en | 5 | 0.90 |

Model size dominates; beam width barely matters. **Caveat:** measured on synthetic
TTS audio, which is unnaturally clean. Real speech should favour `small.en` more
than this suggests — revalidate with your own voice before locking `base.en` in.

### Embeddings without PyTorch

Chroma's built-in ONNX `all-MiniLM-L6-v2` is used instead of `sentence-transformers`:
same model, same vectors, ~80MB via onnxruntime instead of ~2.5GB of torch.

### Expected latency per voice turn

~1.3s STT + 2.7-5.7s agent + near-instant TTS ≈ **4-7s**.

Replies **stream** into the transcript as they generate, so the first words appear in
well under a second rather than after the whole turn. Tool calls are never streamed:
the user should see the result of a lookup, not its JSON.

---

## Using your applications

`launch_app` originally carried seven hardcoded names — notepad, wordpad,
calculator, paint, explorer, snipping tool, character map. Safe, and close to
useless: none of them is an application anyone actually works in.

The fix is **not** to drop the allowlist. It is the whole security story for the
one tool that starts arbitrary programs, and free-form executable names would
make every other guard here decorative. The fix is to make the allowlist *yours*:

* `src/apps.py` discovers what is installed, from Start Menu shortcuts (targets
  resolved in **one** batched PowerShell call — 169 shortcuts is otherwise 169
  process spawns) and from the registry's `App Paths`.
* Duplicates are merged **by target executable**, so the registry's `chrome` and
  the Start Menu's `Google Chrome` are one entry, under the friendlier name.
* Anything resolving to a shell, a system administration tool, or an uninstaller
  is flagged `risky` and **never** enabled by default — including shortcuts whose
  *label* looks harmless. "Anaconda Prompt" and "Node.js command prompt" both
  resolve to `cmd.exe`, and both are caught, because the check reads the target
  rather than the name.
* The **APPS tab** lists all of them; you tick what Jarvis may use. On a first
  run a handful of dull, non-risky suggestions are pre-ticked (editor, browser,
  file manager, office apps) so it is useful immediately. Untick freely.
* Choices persist to `apps.json`.

**Installed is not the same as allowed.** PowerShell is on every Windows machine
and is always in the catalog; `resolve()` returns nothing for it until you
deliberately enable it, and `scripts/test_apps.py` asserts exactly that.

The model supplies a **name**, never a path — the path comes from your allowlist.
A model that cannot name an executable path cannot invent one. An ambiguous name
resolves to nothing rather than to a guess: launching one of two possible apps
because the name was close is worse than saying it was unclear.

`open_with` opens a named file in a named app ("open the budget in Excel"); the
file goes through the same containment as every other path, and executables are
refused as they are for `open_file`. Both tools sit in the destructive tier and
are confirmed every call.

The enabled list is injected into the system prompt, so "open this in VS Code"
costs no round trip to discover that VS Code exists — and a model that can see
the list stops inventing names that are not on it.

---

## Getting a 7B model to actually act

Everything here was measured by running the scenario, not reasoned about. All three
were regressions or failures caught by the live tests, and each is a trap worth
knowing:

**"Act, don't describe" makes it describe.** A system prompt saying *"ACT on requests
rather than describing what could be done"* produced, for "delete the file", **no tool
call and the sentence "Deleting D:\jarvis-sandbox\notes\disposable.txt."** — the model
narrated the action instead of performing it, which is the worst possible outcome: a
confident report of something that never happened. What works is naming the failure
directly: *"A sentence describing an action is not the action. Never say a file was
opened, deleted, created, moved or changed unless a tool has just returned success."*
Restored 3/3 correct tool calls.

**A model with no way to look up a filename will invent one.** Asked to "open my
meeting notes" it tried `notes\my meeting notes.txt`, then `meeting_notes.txt` in the
wrong folder — because `search_files` searches *content* and `list_directory` needs
the folder already. `find_files` closed that gap, and now the whole flow is one hop:
`find_files` → exact path → confirm → open. Prompt wording alone did not fix this;
the missing tool did.

**Error messages are prompts.** `does not exist: <path>` gives a model nothing to
correct with. `_not_found()` now lists what the folder *does* contain, closest names
first, marking directories — so a wrong guess becomes a successful retry instead of a
dead end. An early version listed only files, which reported a root holding just
subfolders as "empty": worse than useless.

**Schema descriptions do the teaching.** Told "always export spreadsheets as CSV, not
XLSX", the model stored the triple `(the user, are_exported_as, spreadsheets)` —
dropping the only word that mattered. Spelling out in the `object` field's description
what a *useful* value looks like, with that exact example, fixed it:
`(the user, prefers, CSV for exported spreadsheets, not XLSX)`.

**An error that lists options gets answered; one that gives an instruction gets
obeyed.** Handed `does not exist` plus four candidate paths, the model reported the
candidates back and asked the user which to use — with the answer already in front
of it. Rewriting the same information as a directive —

```
does not exist: C:\Users\...\Documents\my meeting notes.md
RETRY this same tool call now with path="D:\jarvis-sandbox\notes\meeting_notes.txt"
Other possibilities, if that is the wrong one: ...
```

— turned the dead end into the next tool call. Observed end to end: wrong guess →
retry instruction → correct path → VS Code opened the file → reported truthfully.
That is the self-correction loop, and it is built out of error strings rather than
out of any planner.

**Determinism is not reproducibility.** These cases run at `temperature=0.0`, yet
consecutive runs of the behaviour suite gave 9/9 and then 8/9. Nothing regressed:
the system prompt carries the current time, so the prompt itself differs minute to
minute, and the index shifts as files change. `--runs N` reports a **rate** and
names flaky cases, because a single pass/fail reading of a stochastic system
invents regressions that were never there. Currently 27/27 over 3 runs.

### The bug that made it stop acting, the longer you talked to it

The worst failure in this project so far, and it hid inside the memory feature.

Symptom, from a real session: early turns worked — "draw a rocket" drew one and
opened it in Paint. Later turns did not. "Open excel" answered *"Excel is
opening."* and "open notepad and write about gym at 6" answered *"I have started
writing the document"* — and the audit log recorded **no tool calls at all** for
either. It had stopped acting and started narrating.

The cause was one line of history-keeping:

```python
self.history += [{"role": "user", ...},
                 {"role": "assistant", "content": reply, "tool_calls": []}]
```

Only the prose was kept. The call that produced it was thrown away. So every
past turn in the prompt looked like an assistant *asserting* an outcome with
nothing behind it, and the model did the obvious thing: it copied the pattern.
The failure compounds — each fabrication is saved and teaches the next turn —
which is why it degraded as the conversation went on, and why restarting
"fixed" it until the conversation grew again.

Isolated cleanly. Cold, with no history, `open excel` calls `launch_app` every
time. Replay the exact history from the broken session and the identical words
call nothing:

| | cold | with that history |
|---|---|---|
| `open excel` | `launch_app` ✅ | none — *"Excel is opening."* |
| `open notepad and write about gym at 6` | `launch_app` + `write_file` ✅ | none — invents the note |

History now keeps the turn's **trace**: the tool call, a truncated result, then
the reply. Three consequences worth knowing:

- Trimming cuts on **turn boundaries**, never a flat `[-N:]`. A turn is several
  messages now, and slicing between them leaves a tool result with no call in
  front of it — meaningless to the model, invalid for the chat template.
- `ConversationStore.load()` keeps `tool` rows. It filtered to user/assistant
  only, which would have stripped the evidence back out on every restart and
  quietly reintroduced the whole bug.
- Tool results are truncated to 220 characters in history. What the next turn
  needs is *that the call happened and whether it worked*, not the full body of
  a file search.

Verified over an eight-turn session mixing chat and actions: 5/5 action turns
call a real tool, and the replies claim only what the audit log shows.

### Drawing, and why it is shapes rather than pixels

"Open Paint and draw a rocket" used to fail in the two ways a capability gap
always fails. It launched Paint and handed the job back — *"please draw your
rocket using the tools available"* — and asked to "draw a house and open it in
Paint" it replied *"I have created a simple house and opened it"* having only
launched Paint on a `house.txt` it never wrote. A model with no way to do the
thing will either decline it or claim it.

`draw_image` closes that. The model cannot emit an image, but it can emit
*"triangle here, rectangle below it, two fins, a flame"*, and `src/drawing.py`
renders that shape list to a PNG in an allowed root. `open_in` then shows it
through the same app allowlist `open_with` uses, so one confirmation covers
"draw this and open it in Paint" instead of interrupting twice.

Two things this shook out:

- **A shape with neither fill nor outline is invisible in PIL** — it draws
  nothing, silently. The model omits both routinely, and a house came back as a
  near-blank canvas with only its `line` shapes visible. An uncoloured shape now
  gets a black outline. Drawing something plain beats drawing nothing and
  reporting success.
- **A worked example in the tool description does most of the work.** The rocket
  in the spec is reproduced faithfully. Without one, output collapses to two or
  three shapes.

Honest limit: composition is as good as a 7B placing coordinates. Rockets are
good, houses are wonky, and faces are poor — it will draw a circle and two
triangles and call it a smile. The plumbing is sound; the artist is small.

### The sign-off it would not stop writing

Nearly every turn used to end the same way — "How can I assist you today?",
"Let me know if you need anything else", "Feel free to ask". On screen that is
mild. Spoken aloud it is the *identical sentence after every answer*, and it is
what made the assistant feel dim rather than brisk.

Asking it to stop does not work. The rule went into the system prompt in plain
terms, was verified present in the built prompt, and the model produced the
filler anyway — naming the exact phrases in a prohibition may even prime them.
Worse, those six extra lines sat between "greetings need no tools" and the rule
that *"How is the machine holding up?" is a `system_status` call*, and pushing
those apart **broke tool selection**: behaviour fell to 7/9, with the model
fabricating `{"memory_used": "15GB"}` instead of calling the tool. The prompt is
a crowded place, and adding to it has a cost paid somewhere else.

So the prompt was put back exactly as it was and the trimming happens in code
(`strip_closing_filler`). Two halves, because replies stream:

- **Trim** what is returned — drop trailing sentences that match a sign-off,
  but never empty the reply: "hi" answered with nothing but a pleasantry keeps
  it.
- **Hold** what is streamed — a tail that is still a viable *prefix* of a
  sign-off is withheld, the same trick `_safe_emit_len` already uses for
  `<tool_call>` tags. Once "How can I" is on screen it cannot be taken back, so
  it is never put there. Prose that cannot become filler streams undelayed.

The distinction that took a second pass: `if you need **anything** else` is
filler, `if you need **the Q1 file**, it is on D:` is an answer. Strip the
second and you lose content, so the abstract objects are enumerated rather than
matching `if you need` outright.

---

## Security model

`src/safety.py` is the boundary. A path is allowed only if, **after** resolving
symlinks/junctions and normalising case, it sits inside `ALLOWED_ROOTS`.

**The roots now include real documents**, not just a sandbox:

```
D:\jarvis-sandbox        the original test corpus
~\Documents  ~\Desktop  ~\Downloads
```

Deliberately *not* `C:\` or `D:\` wholesale — containment stops meaning anything
once it covers everything, and `D:\` would include `project-jarvis` itself, letting
the assistant edit its own source and its own app allowlist. Folders that do not
exist are dropped rather than failing, so a machine with OneDrive-redirected profile
folders still starts.

Before widening them, `COMMAND_MODE` was switched from `"denylist"` to
`"allowlist"`, exactly as the module docstring says to. A denylist is a safety net —
bypassable by renaming a binary — which was acceptable when the only reachable files
were throwaway test data and is not acceptable now.

**`INDEX_ROOTS` is a separate, smaller list.** What is *reachable* and what is
*searched by meaning* are different questions, and conflating them measurably hurt
answers — see below.

Verified against: `..` traversal, real Windows directory junctions escaping the root,
sibling-prefix attacks (`docs_evil` vs `docs`), alternate data streams, reserved
device names (`CON`, `NUL`), 8.3 short names, and empty config (fails **closed**).

Enforced in three places — tool dispatch, the index walker, and search results —
because the index is itself a read path into file contents.

**Tiers:** read-only tools auto-execute — `search_files`, `find_files`, `read_file`,
`list_directory`, `remember_fact`, `recall_facts`, the timers, `get_datetime` and
`system_status`. Everything that changes the machine — `write_file`, `edit_file`,
`append_to_file`, `create_directory`, `move_file`, `delete_file`, `run_command`,
`open_file`, `launch_app`, `open_url` — requires explicit UI confirmation **every
call**, with no caching. Closing or Escaping the dialog is a denial, never an
approval.

`open_file` and `launch_app` write nothing, but they start a process on your behalf,
so they sit in the destructive tier too.

The **file browser** in the UI runs its own `ToolExecutor` against the same roots and
the same audit log, so a file previewed by hand is recorded exactly like one the
model read — and it works before the LLM has finished loading.

**Editing notes:** prefer `edit_file` (exact snippet → replacement) over `write_file`.
`write_file` replaces the entire file, so a model changing one line must reproduce
everything else perfectly or it is silently lost. `edit_file` refuses a snippet that
matches zero times *or more than once* — an ambiguous match would edit the wrong line.
The confirmation prompt renders a real unified **diff**, because approving a sentence
that merely claims what will change is not informed consent.

**`open_file` refuses executable types** (`.exe .bat .cmd .ps1 .vbs .lnk .msi …`).
`os.startfile()` on those is arbitrary code execution — containment governs *where* a
file may be, not whether running it is safe. A malicious `.bat` inside an allowed root
is still a malicious `.bat`.

**`launch_app` uses an allowlist** (notepad, wordpad, calculator, paint, explorer,
snipping tool, character map). Free-form executable names would make every other guard
here decorative.

**`run_command`:** never uses a shell. `shell=False` with an argv list, so `|`, `;`,
`>` arrive as inert literal arguments. Denies shell interpreters (`cmd`,
`powershell`, `wscript`) so a shell is never reachable, plus destructive system tools
and network fetchers — matched on the parsed executable name, not raw substrings.
Forced cwd inside a root, 20s timeout.

> **The denylist is a safety net, not a boundary.** Denylists are bypassable by
> renaming a binary. The real protections are no-shell + argv-only + forced cwd +
> timeout + mandatory confirmation. **Set `COMMAND_MODE = "allowlist"` in
> `src/tools.py` before pointing this at real directories.**

**Audit** (`audit.db`, SQLite, WAL): two-phase — intent written *before* execution,
outcome after — so a crash mid-operation still leaves a `pending` row. Denials are
logged too; a rejected traversal attempt is more interesting than a successful read.

### Known limitations

- **TOCTOU:** a validated path could in principle be swapped for a symlink before the
  tool opens it. Acceptable for a single-user local assistant; do not reuse
  `safety.py` multi-tenant without handle-based revalidation.
- **xlsx formulas:** `openpyxl` returns `None` for formula cells with no cached value.
  Excel-saved files are fine; programmatically generated ones lose computed columns.
- **Confirmation shows the model's intent**, which is the right thing to review — but
  read the path in the dialog, not the model's prose summary.

---

## Packaging

```bat
venv\Scripts\pyinstaller.exe packaging\jarvis.spec --noconfirm
```

**PyInstaller over Tauri:** the stack is Python end to end. Tauri would require
shipping Python as a sidecar anyway — keeping every Python packaging problem while
adding a Rust toolchain and IPC. Tauri's small-binary advantage is meaningless when
the payload is gigabytes of CUDA kernels.

**onedir, not onefile:** onefile extracts the whole bundle to temp on *every launch*;
with `ggml-cuda.dll` (703MB) + `cublasLt64_12.dll` (638MB) that is ~1.5GB copied
before the window appears.

**Models are not bundled** (the GGUF alone is 4.47GB) — they download on first run,
which also lets you swap models without rebuilding. Expect ~1.6GB in `dist/Jarvis/`,
almost entirely CUDA. A CPU-only wheel would cut that to ~250MB at ~5-8 tok/s.

---

## Layout

```
main.py                 entry point
src/
  config.py             ALLOWED_ROOTS — the security boundary; INDEX_ROOTS
  safety.py             path containment
  walk.py               one pruning, bounded filesystem walk (see above)
  apps.py               installed-app discovery + the user's launch allowlist
  audit.py              two-phase SQLite audit log
  extract.py            pdf/docx/xlsx/txt text extraction
  index_store.py        Chroma + hybrid dense/lexical retrieval (RRF)
  conversation.py       history that survives a restart, with a staleness rule
  watcher.py            debounced incremental reindexing
  tools.py              tool schema, tiers, dispatch, run_command sandbox
  agent.py              Qwen native tool-calling loop
  knowledge.py          NetworkX personal knowledge graph
  speech.py             wake word / STT / TTS (Piper or SAPI)
  pipeline.py           wires it together; one turn lock over the LLM
  app.py                tkinter UI + tray, thread-safe confirmation bridge
  theme.py              HUD palette, ttk styling, DPI setup, glow/bracket helpers
  reactor.py            animated arc-reactor status ring
scripts/                one verification script per stage
packaging/              PyInstaller spec + CUDA runtime hook
```

## What widening the roots actually broke

Three real defects, invisible while the only root was a 5-file sandbox, all of
which appeared the moment `Downloads` was in scope:

**A 25-second tool call.** `Downloads` holds **229,746 entries** — almost entirely
checked-out repositories — and every `rglob("*")` in the codebase walked all of
them. `find_files` took **25.6s**, against a whole-turn budget of 4–7s. Filtering
after the walk, which the indexer did, does not help: the cost is *descending into*
`node_modules` and `.git`, not the filtering. `src/walk.py` prunes `dirnames` in
place under `os.walk`, so those subtrees are never entered.

| | entries walked | find_files |
|---|---|---|
| before | 232,479 | 25.6s |
| after | 14,853 | **0.1s** |

**Indexing crashed outright.** One document produced 12,715 chunks and Chroma
rejects any single `add()` above its max batch size (5,461 here) — an
`InternalError` that aborted the entire run rather than skipping the one file.
Chunks are now added in batches, and capped per file: one oversized document should
not consume the whole embedding budget or crowd out every other result.

**Answers got worse.** With Downloads indexed, "what is over budget" returned a
README from a cloned project instead of `q1_budget.xlsx`. Downloads is a transit
folder, not a document library. It stays fully *reachable* — `find_files` locates
files there, `read_file` and `open_with` work on them — but it no longer competes
for relevance in content search. The index went from 11,284 chunks back to 1,072,
and the correct answers came back.

`find_files` also weights documents above source files, so "the budget" ranks
`q1_budget.xlsx` above a `budget.py` in some downloaded repo. Ask for `budget.py`
by name and you still get it.

> A consequence worth stating: `scripts/test_index.py` now runs against the sandbox
> in a throwaway Chroma directory. Pointed at the live index it had stopped being a
> regression test — its expected answers depended on whichever files the machine
> happened to hold, so retrieval quality appeared to drift with the user's Desktop
> rather than with the code. `--live` still inspects the real index.

---

## Concurrency

Three locks, each guarding something that a second thread genuinely reached:

| Lock | Guards | The bug without it |
|---|---|---|
| `JarvisApp._pipeline_lock` | building the pipeline | typing a message and pressing ENGAGE during the load built **two** llama.cpp contexts, ~4.5GB each on a 6GB card |
| `JarvisPipeline._turn_lock` | one turn at a time | llama_cpp is not thread-safe; a typed question during a voice reply shared one KV cache and corrupted both |
| `SpeechIO._audio_lock` | the audio device | a timer firing mid-utterance opened playback on top of a live recording stream |

The UI never blocks on any of them: `submit_typed` checks `pipeline.busy` and says so
rather than queueing silently, and a typed turn restores the state the voice loop was
in instead of hard-coding "idle".

## Memory

Three kinds, deliberately separate:

| What | Where | Lifetime |
|---|---|---|
| Facts about you | `knowledge.json` (NetworkX graph) | forever, until forgotten |
| The conversation thread | `conversation.json` | 12 hours, then dropped |
| Document contents | `chroma/` | until the file changes |

The staleness rule on the conversation is the part that matters. Restoring
yesterday's turns into today's prompt is *worse* than restoring nothing — the model
answers a question you have forgotten asking — so history re-enters the prompt only
if the last exchange was recent. Verified end to end: asked "Who is drafting the
Contoso renewal?", killed the process, and a fresh one answered "And when is it
due?" correctly, resolving "it" from restored history.

## Still to do

- **Validate speech with your real voice.** Every STT/wake-word number here comes
  from synthetic Windows SAPI audio.
- Download Piper (above) if you want a better voice than SAPI's.
- Barge-in: `Esc` stops playback, but the wake word is not listened for *while*
  speaking, so you cannot interrupt by voice.
- Window control — focus, list, or close a running app — is not implemented; the
  app tools launch and open, nothing more.
- The first index of a new folder happens on demand, not at startup. Content search
  finds nothing there until then; `find_files` works immediately either way.
