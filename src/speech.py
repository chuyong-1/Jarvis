"""Speech I/O pipeline: wake word -> record -> transcribe -> speak.

Everything here runs on CPU by design. The GPU belongs to the LLM: measured on
this machine, evicting llama.cpp layers to make room for Whisper drops
generation from ~34 tok/s to roughly 10-15, to save ~1.3s of one-shot STT.

Voice output has two local backends: Piper if it has been downloaded, and
Windows' own SAPI otherwise, so the assistant can answer out loud on a clean
machine with nothing fetched. If neither is usable it stays fully functional
and simply does not speak -- a missing TTS voice must never be fatal to a file
assistant.
"""
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLE_RATE = 16000
CHUNK = 1280                    # 80ms, openWakeWord's frame size
WAKE_WORD = "hey_jarvis"
WAKE_THRESHOLD = 0.5
WAKE_DEBOUNCE_SECONDS = 2.0     # one utterance fires ~8 consecutive frames
SILENCE_RMS = 0.012
SILENCE_HANG_SECONDS = 1.2
MAX_UTTERANCE_SECONDS = 15

ROOT = Path(__file__).resolve().parent.parent
PIPER_BIN = ROOT / "piper" / "piper.exe"
PIPER_VOICE = ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
WHISPER_DIR = ROOT / "models" / "whisper"

# Scratch files for the TTS backends, kept out of the user's roots.
VOICE_DIR = ROOT / "models" / "voice"
SPEAK_WAV = VOICE_DIR / "_speak.wav"
SPEAK_TXT = VOICE_DIR / "_speak.txt"

# SAPI tuning. Rate is -10..10; the default reads slightly slow for short
# replies. Voices are tried in order and the first installed one wins --
# "David" is the standard male voice on a UK/US Windows install.
SAPI_RATE = 1
SAPI_VOICE_PREFS = "'Microsoft David Desktop','Microsoft Zira Desktop'"

# Every helper subprocess here is windowless: a console flashing up on screen
# each time the assistant speaks is unacceptable in a packaged app.
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class SpeechIO:
    def __init__(self, whisper_model: str = "base.en", on_state=None, on_level=None):
        self.on_state = on_state or (lambda s: None)
        # Live mic amplitude, so the UI can show that it is really hearing you.
        self.on_level = on_level or (lambda level: None)
        self.whisper_model = whisper_model
        self._oww = None
        self._stt = None
        self._last_wake = 0.0
        self._tts_backend: str | None = None      # probed lazily, then cached
        # One process, one default audio device. Recording and playback must
        # never hold it at the same time -- a timer firing mid-utterance used
        # to open a playback stream on top of a live InputStream.
        self._audio_lock = threading.RLock()
        self._speak_cancel = threading.Event()

    # ---- lazy loading ---------------------------------------------------

    @property
    def wake(self):
        if self._oww is None:
            import openwakeword
            from openwakeword.model import Model
            openwakeword.utils.download_models()
            # Load ONLY the wake word in use. Loading all six pretrained models
            # adds false-positive surface for words the user never says.
            self._oww = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
        return self._oww

    @property
    def stt(self):
        if self._stt is None:
            from faster_whisper import WhisperModel
            # CPU int8 deliberately -- see module docstring.
            self._stt = WhisperModel(self.whisper_model, device="cpu", compute_type="int8",
                                     download_root=str(WHISPER_DIR))
        return self._stt

    # ---- capture --------------------------------------------------------

    def wait_for_wake_word(self, stop_event: threading.Event) -> bool:
        """Block until the wake word is heard. Returns False if stopped."""
        self.on_state("listening")
        model = self.wake
        with self._audio_lock, sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=CHUNK) as stream:
            while not stop_event.is_set():
                frame, _ = stream.read(CHUNK)
                score = float(model.predict(frame.flatten()).get(WAKE_WORD, 0.0))
                now = time.time()
                if score >= WAKE_THRESHOLD and (now - self._last_wake) > WAKE_DEBOUNCE_SECONDS:
                    self._last_wake = now
                    return True
        return False

    def record_utterance(self, stop_event: threading.Event) -> np.ndarray:
        """Record until the speaker goes quiet, or the hard cap is reached."""
        self.on_state("recording")
        frames, silent_for, started = [], 0.0, time.time()
        with self._audio_lock, sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK) as stream:
            while not stop_event.is_set():
                block, _ = stream.read(CHUNK)
                block = block.flatten()
                frames.append(block)
                rms = float(np.sqrt(np.mean(block ** 2)))
                # Scaled so ordinary speech lands near the top of the 0..1 range.
                self.on_level(min(1.0, rms / 0.15))
                silent_for = silent_for + CHUNK / SAMPLE_RATE if rms < SILENCE_RMS else 0.0
                elapsed = time.time() - started
                if silent_for >= SILENCE_HANG_SECONDS and elapsed > 1.0:
                    break
                if elapsed >= MAX_UTTERANCE_SECONDS:
                    break
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        self.on_state("transcribing")
        if audio.size == 0 or float(np.max(np.abs(audio))) < 0.005:
            return ""
        segments, _ = self.stt.transcribe(audio, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    # ---- output ---------------------------------------------------------
    #
    # Two backends, tried in order:
    #
    #   Piper  — better voice, but a manual download (binary + onnx voice).
    #   SAPI   — Windows' own System.Speech, driven through PowerShell. Ships
    #            with the OS, so the assistant can TALK on a clean machine with
    #            nothing downloaded at all.
    #
    # Both are fully local. SAPI existing is why a missing Piper is now a
    # quality regression rather than silence: an assistant you speak to that
    # cannot answer out loud is only half an assistant.
    #
    # INJECTION NOTE: the text spoken is model output, so it is never
    # interpolated into the PowerShell command line. It is written to a UTF-8
    # file that the script reads back. Every path in the command is an
    # app-owned constant.

    def _backend(self) -> str:
        """'piper', 'sapi', or 'none' — cached after the first probe."""
        if self._tts_backend is None:
            if self._piper_ready():
                self._tts_backend = "piper"
            elif self._sapi_ready():
                self._tts_backend = "sapi"
            else:
                self._tts_backend = "none"
        return self._tts_backend

    @staticmethod
    def _piper_ready() -> bool:
        return PIPER_BIN.exists() and PIPER_VOICE.exists() and \
            PIPER_VOICE.with_suffix(".onnx.json").exists()

    @staticmethod
    def _sapi_ready() -> bool:
        """Windows only, and only if a voice is actually installed.

        A Windows install with the speech components stripped has the assembly
        but zero voices, and Speak() then writes an empty wav rather than
        failing — so probe for an installed voice, not just for the platform.
        """
        if sys.platform != "win32":
            return False
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Add-Type -AssemblyName System.Speech; "
                 "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                 ".GetInstalledVoices().Count"],
                capture_output=True, timeout=25,
                creationflags=CREATE_NO_WINDOW)
            return proc.returncode == 0 and int(proc.stdout.decode(errors="replace").strip() or 0) > 0
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def tts_available(cls) -> bool:
        return cls._piper_ready() or cls._sapi_ready()

    def tts_name(self) -> str:
        """What the UI should display for voice output."""
        return {"piper": "PIPER", "sapi": "WINDOWS SAPI", "none": "OFFLINE"}[self._backend()]

    # ---- synthesis ------------------------------------------------------

    def _synth_piper(self, text: str, out: Path) -> bool:
        proc = subprocess.run(
            [str(PIPER_BIN), "--model", str(PIPER_VOICE), "--output_file", str(out)],
            input=text.encode("utf-8"), capture_output=True, timeout=60,
            creationflags=CREATE_NO_WINDOW)
        return proc.returncode == 0 and out.exists()

    def _synth_sapi(self, text: str, out: Path) -> bool:
        SPEAK_TXT.parent.mkdir(parents=True, exist_ok=True)
        SPEAK_TXT.write_text(text, encoding="utf-8")
        # Rate 1 (range -10..10): the default reads a shade slower than is
        # comfortable for short assistant replies.
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {SAPI_RATE}; "
            f"foreach ($v in @({SAPI_VOICE_PREFS})) {{ "
            "  try {{ $s.SelectVoice($v); break }} catch {{}} }}; "
            f"$s.SetOutputToWaveFile('{out}'); "
            f"$s.Speak([System.IO.File]::ReadAllText('{SPEAK_TXT}', "
            "[System.Text.Encoding]::UTF8)); $s.Dispose()"
        ).replace("{{", "{").replace("}}", "}")
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, timeout=60, creationflags=CREATE_NO_WINDOW)
        return proc.returncode == 0 and out.exists() and out.stat().st_size > 44

    # ---- playback -------------------------------------------------------

    def stop_speaking(self) -> None:
        """Cut playback short (barge-in). Safe to call from any thread."""
        self._speak_cancel.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass

    def speak(self, text: str, stop_event: threading.Event | None = None) -> bool:
        """Speak `text`. Returns False (never raises) if no backend is usable.

        Serialised on _audio_lock. The recorder and the player both own the
        default audio device, and a timer firing mid-utterance used to open a
        second stream on top of a live InputStream.
        """
        text = _speakable(text)
        if not text:
            return False
        backend = self._backend()
        if backend == "none":
            return False

        with self._audio_lock:
            self._speak_cancel.clear()
            self.on_state("speaking")
            out = SPEAK_WAV
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                ok = (self._synth_piper(text, out) if backend == "piper"
                      else self._synth_sapi(text, out))
                if not ok:
                    return False
                with wave.open(str(out), "rb") as wf:
                    rate = wf.getframerate()
                    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    if wf.getnchannels() > 1:
                        data = data.reshape(-1, wf.getnchannels())
                sd.play(data, rate)
                # Poll rather than sd.wait() so a stop request or app shutdown
                # can interrupt a long reply instead of blocking on it. The
                # deadline is a backstop: get_stream() reports on the most
                # recent stream and would spin forever if playback never
                # registered as active.
                deadline = time.time() + len(data) / float(rate) + 1.0
                while time.time() < deadline:
                    if self._speak_cancel.is_set() or (stop_event and stop_event.is_set()):
                        sd.stop()
                        return False
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:  # noqa: BLE001 - no stream == nothing playing
                        break
                    time.sleep(0.05)
                return True
            except Exception:  # noqa: BLE001 - TTS must never kill the assistant
                return False


# Markdown and long paths are for the eye, not the ear: read aloud they become
# "asterisk asterisk" and a letter-by-letter drive path. Strip them for speech
# only — the transcript still shows the full text.
_MD_RE = re.compile(r"[*_`#]+")
_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")


def _speakable(text: str, limit: int = 600) -> str:
    text = _PATH_RE.sub(lambda m: Path(m.group(0)).name, text or "")
    text = _MD_RE.sub("", text).strip()
    if len(text) > limit:                      # don't monologue
        cut = text[:limit].rsplit(". ", 1)[0]
        text = (cut + ".") if len(cut) > 80 else text[:limit]
    return text.strip()
