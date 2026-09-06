"""watchdog-based incremental reindexing.

Debouncing is not optional here. Saving a single file in most editors emits a
burst of events (temp file created, renamed over the original, attributes
touched), and Office applications are worse -- a single .docx save can produce
a dozen. Reindexing per event would re-embed the same file repeatedly and, for
a file still mid-write, would read a truncated document. So events are
coalesced per path and only flushed after the path has been quiet.

Containment is re-checked on every event: watchdog reports whatever the OS
tells it, and a junction planted inside a root could otherwise surface paths
from outside it.
"""
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import SUPPORTED_SUFFIXES  # noqa: E402
from safety import is_within  # noqa: E402

DEBOUNCE_SECONDS = 1.5


class _Handler(FileSystemEventHandler):
    def __init__(self, on_change):
        self.on_change = on_change

    def _consider(self, path_str, deleted=False):
        path = Path(path_str)
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            self.on_change(path, deleted)

    def on_created(self, event):
        if not event.is_directory:
            self._consider(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._consider(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._consider(event.src_path, deleted=True)

    def on_moved(self, event):
        if not event.is_directory:
            self._consider(event.src_path, deleted=True)
            self._consider(event.dest_path)


class IndexWatcher:
    """Watches the allowed roots and incrementally updates the index."""

    def __init__(self, index, allowed_roots, debounce=DEBOUNCE_SECONDS, on_event=None):
        self.index = index
        self.roots = allowed_roots
        self.debounce = debounce
        self.on_event = on_event          # optional callback for UI/logging
        self._pending: dict[Path, tuple[float, bool]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer = Observer()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)

    def _queue(self, path: Path, deleted: bool):
        # A path reported by the OS is not automatically inside a root.
        if not deleted and not is_within(path, self.roots):
            return
        with self._lock:
            self._pending[path] = (time.time(), deleted)

    def _flush_loop(self):
        while not self._stop.is_set():
            now = time.time()
            ready = []
            with self._lock:
                for path, (ts, deleted) in list(self._pending.items()):
                    if now - ts >= self.debounce:
                        ready.append((path, deleted))
                        del self._pending[path]
            for path, deleted in ready:
                try:
                    if deleted or not path.exists():
                        self.index.remove_file(path)
                        action = "removed"
                    else:
                        status, n = self.index.index_file(path)
                        action = f"{status}({n})"
                    if self.on_event:
                        self.on_event(path, action)
                except Exception as exc:  # noqa: BLE001 - a watcher must not die
                    if self.on_event:
                        self.on_event(path, f"error: {type(exc).__name__}: {exc}")
            self._stop.wait(0.25)

    def start(self):
        handler = _Handler(self._queue)
        for root in self.roots:
            if Path(root).is_dir():
                self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()
        self._flusher.start()

    def stop(self):
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
