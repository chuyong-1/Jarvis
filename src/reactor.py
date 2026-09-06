"""The animated arc-reactor status ring.

Concentric arc segments rotating at different rates inside an angular frame,
around a pulsing core. The accent colour and rotation speed are driven by the
assistant's state, so the widget is a status indicator first and decoration
second: you can tell from across the room whether it is listening, recording,
or thinking.

Runs on tkinter's after() loop on the MAIN thread only. It must never be
driven from the worker thread.

DRAWING BUDGET -- every item is redrawn every frame, so the item count IS the
frame budget: roughly 180 primitives here, which tkinter's canvas handles
comfortably on the machine this targets. Anything constant across frames (the
tick ring's angles, the hexagonal frame) is computed once in __init__ rather
than 30 times a second.
"""
import math
import tkinter as tk

import theme

FRAME_MS = 33          # ~30fps
EASE = 0.18            # per-frame colour approach; settles in ~200ms


class Reactor(tk.Canvas):
    def __init__(self, master, size=190, **kw):
        super().__init__(master, width=size, height=size, bg=theme.BG,
                         highlightthickness=0, **kw)
        self.size = size
        self.angle = 0.0
        self.pulse = 0.0
        self.state = "idle"
        self.level = 0.0          # 0..1 live mic level, drives the inner bars
        self._running = True
        # Eased colour: the ring fades between states rather than snapping. A
        # hard cut from cyan to red reads as a glitch; a short fade reads as a
        # transition.
        self._colour = theme.STATES["idle"][0]
        self._from_colour = self._colour
        self._ease = 1.0
        c = size / 2
        self._hex = [(c + (c - 2) * math.cos(math.radians(a)),
                      c + (c - 2) * math.sin(math.radians(a)))
                     for a in range(-90, 270, 60)]
        self._ticks = [(math.cos(math.radians(i * 6)), math.sin(math.radians(i * 6)),
                        i % 5 == 0) for i in range(60)]
        self.after(FRAME_MS, self._tick)

    def set_state(self, state):
        state = state if state in theme.STATES else "idle"
        if state != self.state:
            self._from_colour = self._colour      # ease from wherever we are now
            self._ease = 0.0
        self.state = state

    def set_level(self, level):
        self.level = max(0.0, min(1.0, level))

    def stop(self):
        self._running = False

    # ---- drawing ---------------------------------------------------------

    def _tick(self):
        if not self._running:
            return
        target, _label, speed = theme.STATES.get(self.state, theme.STATES["idle"])
        self._ease = min(1.0, self._ease + EASE)
        self._colour = theme.mix(self._from_colour, target, self._ease)
        self.angle = (self.angle + 2.2 * speed) % 360
        self.pulse = (self.pulse + 0.06 * max(speed, 0.3)) % (2 * math.pi)
        self._draw(self._colour, speed)
        self.after(FRAME_MS, self._tick)

    def _draw(self, colour, speed):
        self.delete("all")
        c = self.size / 2
        breathe = (math.sin(self.pulse) + 1) / 2            # 0..1
        dim = theme.blend(colour, theme.BG, 0.72)

        # Angular frame: a hexagon rather than a circle, so the widget reads as
        # instrumentation instead of a loading spinner.
        self.create_polygon(self._hex, outline=theme.blend(colour, theme.BG, 0.86),
                            fill="", width=1)

        # Outer tick marks -- a fixed reference the moving rings read against.
        r_ticks = c - 10
        for cos_a, sin_a, long in self._ticks:
            r0 = r_ticks - (7 if long else 3)
            self.create_line(c + r0 * cos_a, c + r0 * sin_a,
                             c + r_ticks * cos_a, c + r_ticks * sin_a,
                             fill=theme.blend(colour, theme.BG, 0.55 if long else 0.8),
                             width=1)

        # Radar sweep: a short leading arc with a fading tail. Conveys direction
        # of travel, which a symmetric ring cannot.
        sweep_r = c - 14
        for i in range(7):
            self.create_arc(c - sweep_r, c - sweep_r, c + sweep_r, c + sweep_r,
                            start=(-self.angle * 1.4 - i * 7) % 360, extent=7,
                            style="arc", width=2,
                            outline=theme.blend(colour, theme.BG, 0.25 + i * 0.11))

        # Three counter-rotating arc rings.
        for radius, extent, direction, width in ((c - 24, 250, 1.0, 3),
                                                 (c - 38, 140, -1.7, 2),
                                                 (c - 51, 300, 2.4, 2)):
            start = (self.angle * direction) % 360
            # Fake a glow: the same arc drawn wider and dimmer underneath.
            for pad, fade in ((width + 4, 0.82), (width + 2, 0.6), (0, 0.0)):
                self.create_arc(c - radius, c - radius, c + radius, c + radius,
                                start=start, extent=extent, style="arc",
                                outline=theme.blend(colour, theme.BG, fade),
                                width=width + pad)

        # Markers orbiting the outer ring.
        for i in range(3):
            a = math.radians(self.angle * 1.9 + i * 120)
            r = c - 24
            x, y = c + r * math.cos(a), c + r * math.sin(a)
            self.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5,
                             outline="", fill=theme.blend(colour, theme.BG, 0.15))

        # Inner level bars -- live mic amplitude while recording, breathing otherwise.
        r_in = c - 60
        bars = 36
        amplitude = self.level if self.state == "recording" else breathe * 0.45
        for i in range(bars):
            a = math.radians(i * (360 / bars) - 90)
            # Deterministic per-bar variation so it looks like a spectrum, not a circle.
            wobble = (math.sin(self.pulse * 1.7 + i * 0.7) + 1) / 2
            length = 4 + amplitude * 18 * (0.45 + 0.55 * wobble)
            self.create_line(
                c + r_in * math.cos(a), c + r_in * math.sin(a),
                c + (r_in - length) * math.cos(a), c + (r_in - length) * math.sin(a),
                fill=theme.blend(colour, theme.BG, 0.35), width=2)

        # Core.
        r_core = 16 + breathe * 4
        for pad, fade in ((10, 0.85), (6, 0.7), (3, 0.45), (0, 0.0)):
            r = r_core + pad
            self.create_oval(c - r, c - r, c + r, c + r,
                             outline="", fill=theme.blend(colour, theme.BG, fade))
        # Crosshair through the core: the detail that reads "instrument" rather
        # than "progress spinner".
        self.create_line(c - r_core - 8, c, c - r_core, c, fill=dim, width=1)
        self.create_line(c + r_core, c, c + r_core + 8, c, fill=dim, width=1)
        self.create_oval(c - 5, c - 5, c + 5, c + 5, outline="", fill="#eaffff")
