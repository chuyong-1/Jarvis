"""Timers and reminders.

Deliberately NOT implemented as run_command("timeout /t 30"), which is what the
model reaches for when no timer tool exists. That approach is broken three ways:
it blocks the assistant for the whole duration, it needs a confirmation click
for something trivial, and run_command's 20s cap would kill any longer timer
part-way through -- silently, so the timer simply never fires.

A real timer is a background threading.Timer that fires a callback. Timers live
in memory only: they do not survive an app restart, which is the honest
trade-off for not shipping a scheduler daemon. Anything that must survive a
reboot belongs in Windows Task Scheduler, not here.
"""
import itertools
import threading
import time


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


class Timer:
    def __init__(self, timer_id: int, seconds: float, label: str, handle: threading.Timer):
        self.id = timer_id
        self.seconds = seconds
        self.label = label
        self.handle = handle
        self.created = time.time()
        self.fired = False
        self.cancelled = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.created + self.seconds - time.time())


class TimerManager:
    """Schedules timers and reports when they fire, via on_fire(timer)."""

    def __init__(self, on_fire=None):
        self.on_fire = on_fire or (lambda timer: None)
        self._timers: dict[int, Timer] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def set_timer(self, seconds: float, label: str = "") -> Timer:
        if seconds <= 0:
            raise ValueError("duration must be positive")
        timer_id = next(self._ids)

        def fire():
            with self._lock:
                timer = self._timers.get(timer_id)
                if timer is None or timer.cancelled:
                    return
                timer.fired = True
            # Callback runs OUTSIDE the lock: it reaches into the UI and may
            # block on speech, and holding the lock would stall other timers.
            self.on_fire(timer)

        handle = threading.Timer(seconds, fire)
        handle.daemon = True
        timer = Timer(timer_id, seconds, label, handle)
        with self._lock:
            self._timers[timer_id] = timer
        handle.start()
        return timer

    def cancel(self, timer_id: int) -> bool:
        with self._lock:
            timer = self._timers.get(timer_id)
            if timer is None or timer.fired or timer.cancelled:
                return False
            timer.cancelled = True
            timer.handle.cancel()
            return True

    def active(self) -> list[Timer]:
        with self._lock:
            return [t for t in self._timers.values() if not (t.fired or t.cancelled)]

    def cancel_all(self) -> None:
        for timer in self.active():
            self.cancel(timer.id)
