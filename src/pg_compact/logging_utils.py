"""Plain, timestamped, line-oriented logging.

Deliberately avoids ANSI cursor-movement escapes (\\e[1A, \\e[K, carriage
return redraws) used by the predecessor shell script's progress bar. Those
only render correctly in an interactive terminal; redirected to a file, piped
through `tee`, or captured by cron/systemd they leave behind garbled escape
sequences or a wall of overwritten-looking lines. Emitting one plain line per
event works identically in a terminal, a log file, or `journalctl`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from enum import IntEnum
from typing import Callable

LogFn = Callable[[str, str], None]  # (level, message)


def noop_log(level: str, message: str) -> None:
    """Default no-op logger for callers that don't care about output."""


def fmt_bytes(n: int | None) -> str:
    """Human-readable byte size, e.g. ``5.5 GB``.  Returns ``?`` for None."""
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.0f} {unit}" if value == int(value) else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def fmt_size_change(before: int, after: int) -> str:
    """Human-readable size change: ``freed 42.6%``, ``grew 0.2%``, ``no change``.

    Uses a plain verb instead of a signed percentage so a reduction can
    never be misread as growth (a bare ``+42.6%`` next to ``376 KB ->
    216 KB`` looks like the table got bigger).  The magnitude is the
    absolute change relative to the starting size.
    """
    if before <= 0 or before == after:
        return "no change"
    pct = 100.0 * abs(after - before) / before
    verb = "freed" if after < before else "grew"
    return f"{verb} {pct:.1f}%"


def fmt_duration(seconds: float) -> str:
    """Human-readable duration: '1h 36m', '12m 5s', '45s'.  '-' for non-positive.

    Uses the two most significant units so the value reads like a clock
    time (e.g. '1h 36m') rather than a fraction ('1.6h').
    """
    if seconds <= 0:
        return "-"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    if minutes:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    return f"{secs}s"


class Level(IntEnum):
    # Ordering is the filter priority: a message shows when its level is
    # >= the logger's verbosity.  NOTICE ranks just above INFO so milestone
    # notices (phase-skip reasons, stop reasons, TOAST strategy, chunk-map
    # build) are visible at the default level and in log files, but still
    # suppressed by -q.  Only -v drops to DEBUG (the per-round status spam).
    DEBUG = -1
    INFO = 0
    NOTICE = 1
    WARNING = 2
    ERROR = 3
    ALWAYS = 100


_LEVEL_BY_NAME = {
    "debug": Level.DEBUG,
    "notice": Level.NOTICE,
    "info": Level.INFO,
    "warning": Level.WARNING,
    "error": Level.ERROR,
    "always": Level.ALWAYS,
}


class Logger:
    def __init__(self, verbosity: Level = Level.INFO, stream=None) -> None:
        self.verbosity = verbosity
        self.stream = stream or sys.stderr
        # Force UTF-8 on the output stream.  On Windows the console/pipe
        # stream defaults to the locale codepage (e.g. cp1251), which
        # mangles any non-ASCII byte written to a log file or through a
        # pipe.  Reconfiguring guarantees consistent UTF-8 output whether
        # the log goes to a terminal, a file, or a redirect.
        reconfigure = getattr(self.stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                # Stream doesn't support reconfiguration (already detached,
                # or a non-text wrapper) - leave it as-is.
                pass

    def log(self, level_name: str, message: str) -> None:
        level = _LEVEL_BY_NAME.get(level_name, Level.INFO)
        if level < self.verbosity:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tag = level_name.upper()
        print(f"[{timestamp}] {tag}: {message}", file=self.stream, flush=True)

    def __call__(self, level_name: str, message: str) -> None:
        self.log(level_name, message)


def verbosity_from_flags(verbose: bool, quiet: bool) -> Level:
    if quiet:
        return Level.ERROR
    if verbose:
        return Level.DEBUG  # -v shows everything, including debug status lines
    return Level.INFO
